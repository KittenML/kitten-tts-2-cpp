#include "engine.h"
#include "llama.h"
#include "text.h"
#include "sampling.h"
#include "decoder.h"
#include <chrono>
#include <fstream>
#include <mutex>
#include <stdexcept>

namespace kitten {
using clock_type = std::chrono::steady_clock;
static json read_json(const std::string & path) {
    std::ifstream f(path); if(!f) throw std::runtime_error("cannot open "+path); json j; f>>j;return j;
}
// LibTorch decoder threads and random state are process-wide.
static std::mutex inference_mutex;
static std::vector<int> tokenize(const llama_vocab * vocab,const std::string & text) {
    int n=llama_tokenize(vocab,text.data(),text.size(),nullptr,0,false,true);
    if(n==0) return {};
    std::vector<int> ids(-n); n=llama_tokenize(vocab,text.data(),text.size(),ids.data(),ids.size(),false,true);
    if(n<0) throw std::runtime_error("tokenization failed");
    ids.resize(n);return ids;
}
struct batch {
    llama_batch b;
    batch(int n,int dim=0): b(llama_batch_init(n,dim,1)) {}
    ~batch(){llama_batch_free(b);}
    void position(int i,int pos,bool logits) {b.pos[i]=pos;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=logits;}
};

struct engine::impl {
    std::map<std::string, std::string> defaults;
    kitten::assets assets;
    json config, voices;
    std::unique_ptr<kitten_text_processing::Normalizer> normalizer;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<kitten::decoder> decoder;
};
engine::engine(const std::map<std::string, std::string> & args): p(std::make_unique<impl>()) {
    std::lock_guard<std::mutex> lock(inference_mutex);
    auto get=[&](const std::string & key,const std::string & fallback){auto it=args.find(key);return it==args.end()?fallback:it->second;};
    p->defaults=args;
    p->defaults.erase("--hf-token");
    int threads=std::stoi(get("--threads","8")), decoder_threads=std::stoi(get("--decoder-threads",std::to_string(threads)));
    if(threads<1 || decoder_threads<1) throw std::invalid_argument("thread counts must be positive");
    p->assets=resolve_assets(args);
    p->config=p->assets.config;
    p->voices=read_json(p->assets.voices);
    p->normalizer=std::make_unique<kitten_text_processing::Normalizer>(get("--data",KITTEN_DEFAULT_DATA));
    static std::once_flag initialized;
    std::call_once(initialized, [] { ggml_backend_load_all(); llama_backend_init(); });
    if(!args.count("--decode-tokens")) {
        auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=!args.count("--no-repack");
        p->model.reset(llama_model_load_from_file(p->assets.model.c_str(),mp));
        if(!p->model) throw std::runtime_error("cannot load language model");
        token_map tm(p->config.at("token_map"));
        if(tm.base+tm.count>llama_vocab_n_tokens(llama_model_get_vocab(p->model.get()))) throw std::runtime_error("token map exceeds vocabulary");
    }
    if(!p->assets.decoder.empty()) p->decoder=std::make_unique<kitten::decoder>(p->assets.decoder,decoder_threads,0);
}
engine::~engine() = default;
json engine::metadata() const {
    json names=json::array(); for(auto it=p->voices.begin();it!=p->voices.end();++it) names.push_back(it.key());
    return {{"config",p->config},{"voices",names}};
}
std::string engine::normalize(const std::string & text) {
    std::lock_guard<std::mutex> lock(inference_mutex);
    return kitten::normalize(text,*p->normalizer,true);
}
generation_result engine::generate(const std::map<std::string, std::string> & options,
    const std::function<bool(const std::vector<float> &)> & on_chunk) {
    std::lock_guard<std::mutex> lock(inference_mutex);
    auto args=p->defaults; for(const auto & pair:options) args[pair.first]=pair.second;
    auto get=[&](const std::string & key,const std::string & fallback){auto it=args.find(key);return it==args.end()?fallback:it->second;};
    const auto & config=p->config;
    const auto & voices=p->voices;
    auto & model=p->model;
    auto & decoder=p->decoder;
    auto & normalizer=*p->normalizer;
    token_map tm(config.at("token_map"));
    std::string voice=get("--voice",config.value("default_voice","Bruno"));
    if(!voices.contains(voice)) throw std::invalid_argument("unknown voice: "+voice);
    const auto & ref=voices.at(voice);
    int threads=std::stoi(get("--threads","8"));
    int seed=std::stoi(get("--seed",std::to_string(std::random_device{}() % 2147483647))), max_tokens=std::stoi(get("--max-tokens","1000"));
    if(threads<1 || max_tokens<1) throw std::invalid_argument("thread and token counts must be positive");
    if(args.count("--decode-tokens")) {
        auto tokens=read_json(args.at("--decode-tokens")).get<std::vector<int>>();
        tokens.erase(std::remove_if(tokens.begin(),tokens.end(),[&](int id){return id<0 || id>=tm.count;}),tokens.end());
        if(tokens.empty()) throw std::runtime_error("no valid codec tokens");
        if(!decoder) throw std::runtime_error("decoder is not loaded");
        decoder->seed(seed);
        auto begin=clock_type::now();
        auto audio=decoder->decode(tokens,ref.dump());
        json report={{"samples",audio.size()},{"decoder_seconds",std::chrono::duration<double>(clock_type::now()-begin).count()}};
        return {std::move(audio),std::move(report)};
    }
    if(!model) throw std::runtime_error("language model is not loaded");
    auto gen=config.value("generation",json::object());
    int chunk_chars=std::stoi(get("--chunk-chars",std::to_string(gen.value("chunk_chars",380))));
    int chunk_min=std::stoi(get("--chunk-min-chars",std::to_string(gen.value("chunk_min_chars",130))));
    float gap=std::stof(get("--chunk-gap","0.16"));
    if(!std::isfinite(gap) || gap<0 || gap>60) throw std::runtime_error("invalid chunk gap");
    auto spoken=kitten::normalize(args.at("--text"),normalizer,!args.count("--no-normalize"));
    auto chunks=kitten::split(spoken,chunk_chars,chunk_min);
    if(chunks.empty()) throw std::runtime_error("no text to speak");
    auto preset=config.at("decode_presets").at(get("--preset",config.value("default_preset","stable")));
    kitten::sampling settings;
    settings.temperature=std::stof(get("--temperature",std::to_string(preset.at("temperature").get<float>())));
    settings.top_k=std::stoi(get("--top-k",std::to_string(preset.at("top_k").get<int>())));
    settings.top_p=std::stof(get("--top-p",std::to_string(preset.at("top_p").get<float>())));
    settings.min_p=std::stof(get("--min-p",std::to_string(preset.value("min_p",0.f))));
    settings.repetition=std::stof(get("--repetition-penalty","1.1"));settings.window=std::stoi(get("--repetition-window",std::to_string(gen.value("repetition_window",0))));
    settings.run_penalty=std::stof(get("--run-penalty","1.3"));settings.grace=std::stoi(get("--run-grace","10"));
    if(!std::isfinite(settings.temperature) || settings.temperature<=0 || settings.top_k<0 || !(settings.top_p>0 && settings.top_p<=1) || !(settings.min_p>=0 && settings.min_p<=1) || !std::isfinite(settings.repetition) || settings.repetition<=0 || !std::isfinite(settings.run_penalty) || settings.run_penalty<0 || settings.grace<1) throw std::runtime_error("invalid sampling settings");
    auto vocab=llama_model_get_vocab(model.get());int n_vocab=llama_vocab_n_tokens(vocab);
    auto speaker=ref.at("speaker").get<std::vector<float>>();
    if(speaker.size()!=size_t(llama_model_n_embd(model.get()))) throw std::runtime_error("speaker projection dimension mismatch");
    auto ref_text=tokenize(vocab,ref.at("transcript").get<std::string>());
    auto ref_audio=ref.at("reference_tokens").get<std::vector<int>>();
    std::vector<int> emotion;
    if((args.count("--use-emotion") ? args.at("--use-emotion")=="1" : kitten::has_expression(args.at("--text"))) && gen.contains("emotion_control")) emotion=tokenize(vocab,gen.at("emotion_control"));
    std::mt19937 rng(seed);
    std::vector<std::vector<float>> waves;
    json report={{"normalized",spoken},{"voice",voice},{"sample_rate",24000},{"chunks",json::array()}};
    std::ofstream logits_file;
    if(args.count("--logits")) {logits_file.open(args.at("--logits"),std::ios::binary);if(!logits_file) throw std::runtime_error("cannot open logits output");}
    std::vector<int> forced;
    if(args.count("--force-tokens")) {forced=read_json(args.at("--force-tokens")).get<std::vector<int>>();if(chunks.size()!=1) throw std::runtime_error("teacher forcing requires one chunk");}
    report["iteration"]=1; report["seed"]=seed;
    if(decoder) decoder->seed(seed);
    double lm_seconds=0,decoder_seconds=0;size_t total_tokens=0;
    for(auto & chunk:chunks) {
        auto ids=kitten::prompt(tm,tokenize(vocab,chunk),ref_text,ref_audio,emotion,args.count("--use-reference") ? args.at("--use-reference")=="1" : (!args.count("--no-reference") && gen.value("use_reference_prompt",true)));
        int budget=chunks.size()==1?max_tokens:std::max(200,std::min(max_tokens,int(kitten::characters(chunk)/20.0*25*1.8)));
        if(args.count("--force-tokens")) budget=forced.size()+1;
        auto cp=llama_context_default_params();cp.n_ctx=ids.size()+budget;cp.n_batch=512;cp.n_ubatch=512;cp.n_threads=threads;cp.n_threads_batch=threads;cp.no_perf=false;
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        if(!ctx) throw std::runtime_error("cannot initialize context");
        auto begin=clock_type::now();
        { batch b(1,speaker.size());b.b.n_tokens=1;std::copy(speaker.begin(),speaker.end(),b.b.embd);b.position(0,0,false);
          if(llama_decode(ctx.get(),b.b)) throw std::runtime_error("speaker prefill failed"); }
        for(size_t offset=1;offset<ids.size();) {
            int n=std::min(size_t(512),ids.size()-offset);batch b(n);b.b.n_tokens=n;
            for(int i=0;i<n;++i) {b.b.token[i]=ids[offset+i];b.position(i,offset+i,offset+i+1==ids.size());}
            if(llama_decode(ctx.get(),b.b)) throw std::runtime_error("prompt prefill failed");
            offset+=n;
        }
        auto history=ids;std::vector<int> audio_ids,generated;bool ended=false;
        for(int step=0;step<budget;++step) {
            const float * raw=llama_get_logits_ith(ctx.get(),-1);
            if(logits_file.is_open()) logits_file.write(reinterpret_cast<const char *>(raw),n_vocab*sizeof(float));
            if(args.count("--force-tokens") && step==int(forced.size())) break;
            std::vector<float> scores(raw,raw+n_vocab);kitten::process(scores,history,ids.size(),tm,settings);
            int token=args.count("--force-tokens")?forced.at(step):kitten::sample(scores,rng,args.count("--greedy"));
            if(token<0 || token>=n_vocab) throw std::runtime_error("forced token out of range");
            generated.push_back(token);history.push_back(token);
            if(!args.count("--force-tokens") && (token==tm.speech_end || token==tm.stop)) {ended=true;break;}
            if(token>=tm.base && token<tm.base+tm.count) audio_ids.push_back(token-tm.base);
            if(step+1<budget) {batch b(1);b.b.n_tokens=1;b.b.token[0]=token;b.position(0,history.size()-1,true);if(llama_decode(ctx.get(),b.b)) throw std::runtime_error("generation failed");}
        }
        double elapsed=std::chrono::duration<double>(clock_type::now()-begin).count();lm_seconds+=elapsed;total_tokens+=generated.size();
        report["chunks"].push_back({{"text",chunk},{"prompt",ids},{"generated",generated},{"audio_tokens",audio_ids},{"terminated",ended},{"lm_seconds",elapsed}});
        llama_perf_context_print(ctx.get());
        if(decoder) {begin=clock_type::now();auto wave=decoder->decode(audio_ids,ref.dump());decoder_seconds+=std::chrono::duration<double>(clock_type::now()-begin).count(); if(on_chunk && !wave.empty() && !on_chunk(wave)) throw std::runtime_error("generation cancelled"); waves.push_back(std::move(wave));}
    }
    report["lm_seconds"]=lm_seconds;report["decoder_seconds"]=decoder_seconds;report["generated_tokens"]=total_tokens;
    generation_result result;
    if(decoder) {result.audio=kitten::join(waves,gap);report["audio_seconds"]=result.audio.size()/24000.0;report["rtf"]=(lm_seconds+decoder_seconds)/(result.audio.size()/24000.0);}
    result.report=std::move(report);
    return result;
}
}
