#include "llama.h"
#include "log.h"
#include "text.h"
#include "sampling.h"
#include "decoder.h"
#include "assets.h"
#include <chrono>
#include <clocale>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
using kitten::json;
using clock_type = std::chrono::steady_clock;
static json read_json(const std::string & path) {
    std::ifstream f(path); if(!f) throw std::runtime_error("cannot open "+path); json j; f>>j;return j;
}
static void write_json(const std::string & path,const json & j) { std::ofstream f(path); f<<j.dump(2)<<'\n';if(!f) throw std::runtime_error("cannot write "+path); }
static void write_wav(const std::string & path,const std::vector<float> & audio) {
    std::ofstream f(path,std::ios::binary);
    auto u16=[&](uint16_t v){ f.put(v&255);f.put(v>>8); };
    auto u32=[&](uint32_t v){ for(int i=0;i<4;++i) f.put((v>>(i*8))&255); };
    f.write("RIFF",4);u32(36+audio.size()*4);f.write("WAVEfmt ",8);u32(16);u16(3);u16(1);u32(24000);u32(96000);u16(4);u16(32);
    f.write("data",4);u32(audio.size()*4);
    for(float x:audio) {uint32_t bits; std::memcpy(&bits,&x,4);u32(bits);}
    if(!f) throw std::runtime_error("cannot write "+path);
}
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
int main(int argc,char ** argv) try {
    std::setlocale(LC_ALL,"C.UTF-8");
    try { std::locale::global(std::locale("C.UTF-8")); } catch (const std::runtime_error &) {}
    std::map<std::string,std::string> args;
    const std::vector<std::string> flags={"--download-only","--offline","--no-normalize","--no-reference","--no-repack","--greedy","--tokens-only","--frontend","--help","--quiet"};
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(std::find(flags.begin(),flags.end(),key)!=flags.end()) args[key]="1";
        else {if(key.rfind("--",0)!=0 || i+1>=argc) throw std::runtime_error("expected --option value");args[key]=argv[++i];}
    }
    if(args.count("--help") || argc==1) {
        std::cout<<"KittenTTS2 CPU inference\n"
          "  kitten-tts --text TEXT [--voice Bruno] [--output output.wav]\n"
          "  --repo ID --revision REV --decoder default|student_w4|student_w8\n"
          "  --cache-dir DIR --offline --download-only\n"
          "  --assets DIR   use local assets instead of downloading\n"
          "  --model FILE   override the GGUF with a local file\n"
          "  --data DIR      prepared kitten-text-processing grammar directory\n"
          "  --threads N --decoder-threads N --seed N --preset stable|expressive\n"
          "  --no-repack    disable CPU weight repacking to reduce runtime memory\n"
          "  --max-tokens N --temperature F --top-k N --top-p F --min-p F\n"
          "  --repetition-penalty F --repetition-window N --run-penalty F --run-grace N\n"
          "  --chunk-chars N --chunk-min-chars N --chunk-gap F\n"
          "  --no-normalize --no-reference --greedy --tokens-only --report FILE\n"
          "  --logits FILE --force-tokens FILE (teacher-forced LM parity)\n"
          "  --decode-tokens FILE (zero-based codec IDs) --repeat N (benchmark)\n"
          "  --frontend (JSON lines on stdin; text/chunking/sampling diagnostics)\n"
          "  --quiet        show warnings, errors and a saved-file message\n";
        return 0;
    }
    auto get=[&](const std::string & key,const std::string & fallback){auto it=args.find(key);return it==args.end()?fallback:it->second;};
    const std::vector<std::string> known={"--repo","--revision","--decoder","--cache-dir","--assets","--model","--text","--voice","--output","--data","--threads","--decoder-threads","--seed","--preset","--max-tokens","--temperature","--top-k","--top-p","--min-p","--repetition-penalty","--repetition-window","--run-penalty","--run-grace","--chunk-chars","--chunk-min-chars","--chunk-gap","--report","--logits","--force-tokens","--decode-tokens","--repeat"};
    for(auto & a:args) if(std::find(flags.begin(),flags.end(),a.first)==flags.end() && std::find(known.begin(),known.end(),a.first)==known.end()) throw std::runtime_error("unknown option "+a.first);
    if(args.count("--quiet")) {
        common_log_set_verbosity_thold(LOG_LEVEL_WARN);
        llama_log_set(common_log_default_callback,nullptr);
        std::atexit([](){common_log_flush(common_log_main());});
    }
    kitten_text_processing::Normalizer normalizer(get("--data",KITTEN_DEFAULT_DATA));
    if(args.count("--frontend")) {
        std::string line;
        while(std::getline(std::cin,line)) {
            try {
                auto j=json::parse(line);
                if (j.value("operation", "") == "join") {
                    auto waves=j.at("waves").get<std::vector<std::vector<float>>>();
                    std::cout<<json{{"audio",kitten::join(waves,j.value("gap",0.16f))}}.dump()<<std::endl;
                    continue;
                }
                if (j.value("operation", "") == "sampling") {
                    auto scores=j.at("scores").get<std::vector<float>>();
                    auto history=j.at("history").get<std::vector<int>>();
                    kitten::token_map t(j.at("token_map")); kitten::sampling opts;
                    auto o=j.at("settings");
                    opts.temperature=o.at("temperature"); opts.top_k=o.at("top_k");
                    opts.top_p=o.at("top_p"); opts.min_p=o.at("min_p");
                    opts.repetition=o.at("repetition_penalty"); opts.window=o.value("repetition_window",0);
                    opts.grace=o.at("token_run_grace"); opts.run_penalty=o.at("token_run_penalty");
                    kitten::process(scores,history,j.at("prompt_length"),t,opts);
                    std::cout<<json{{"scores",scores}}.dump()<<std::endl;
                    continue;
                }
                std::string text=j.value("text","");
                auto normalized=kitten::normalize(text,normalizer,j.value("normalize",true));
                std::cout<<json{{"normalized",normalized},{"chunks",kitten::split(normalized,j.value("chunk_chars",380),j.value("chunk_min_chars",130))},{"expression",kitten::has_expression(text)}}.dump()<<std::endl;
            } catch(const std::exception & e) {std::cout<<json{{"error",e.what()}}.dump()<<std::endl;}
        }
        return 0;
    }
    if(!args.count("--download-only") && !args.count("--text") && !args.count("--decode-tokens")) throw std::runtime_error("--text or --decode-tokens is required");
    auto assets=kitten::resolve_assets(args);
    if(args.count("--download-only")) { std::cout<<"Assets ready\n"; return 0; }
    auto config=assets.config;
    kitten::token_map tm(config.at("token_map"));
    auto voices=read_json(assets.voices);
    std::string voice=get("--voice",config.value("default_voice","Bruno"));
    if(!voices.contains(voice)) throw std::runtime_error("unknown voice: "+voice);
    const auto & ref=voices.at(voice);
    int threads=std::stoi(get("--threads","8")), decoder_threads=std::stoi(get("--decoder-threads",std::to_string(threads)));
    int seed=std::stoi(get("--seed",std::to_string(std::random_device{}() % 2147483647))), max_tokens=std::stoi(get("--max-tokens","1000"));
    if(threads<1 || decoder_threads<1 || max_tokens<1) throw std::runtime_error("thread and token counts must be positive");
    if (args.count("--decode-tokens")) {
        auto tokens=read_json(args.at("--decode-tokens")).get<std::vector<int>>();
        tokens.erase(std::remove_if(tokens.begin(),tokens.end(),[&](int id){return id<0 || id>=tm.count;}),tokens.end());
        if(tokens.empty()) throw std::runtime_error("no valid codec tokens");
        kitten::decoder dec(assets.decoder,decoder_threads,seed);
        auto begin=clock_type::now();
        auto audio=dec.decode(tokens,ref.dump());
        write_wav(get("--output","output.wav"),audio);
        std::cout<<json{{"samples",audio.size()},{"decoder_seconds",std::chrono::duration<double>(clock_type::now()-begin).count()}}.dump()<<'\n';
        return 0;
    }
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
    ggml_backend_load_all();llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=!args.count("--no-repack");
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(assets.model.c_str(),mp),llama_model_free);
    if(!model) throw std::runtime_error("cannot load language model");
    auto vocab=llama_model_get_vocab(model.get());int n_vocab=llama_vocab_n_tokens(vocab);
    if(tm.base+tm.count>n_vocab) throw std::runtime_error("token map exceeds vocabulary");
    auto speaker=ref.at("speaker").get<std::vector<float>>();
    if(speaker.size()!=size_t(llama_model_n_embd(model.get()))) throw std::runtime_error("speaker projection dimension mismatch");
    auto ref_text=tokenize(vocab,ref.at("transcript").get<std::string>());
    auto ref_audio=ref.at("reference_tokens").get<std::vector<int>>();
    std::vector<int> emotion;
    if(kitten::has_expression(args.at("--text")) && gen.contains("emotion_control")) emotion=tokenize(vocab,gen.at("emotion_control"));
    std::mt19937 rng(seed);
    std::unique_ptr<kitten::decoder> decoder;
    if(!args.count("--tokens-only")) decoder=std::make_unique<kitten::decoder>(assets.decoder,decoder_threads,seed);
    std::vector<std::vector<float>> waves;
    json report={{"normalized",spoken},{"voice",voice},{"sample_rate",24000},{"chunks",json::array()}};
    std::ofstream logits_file;
    if(args.count("--logits")) {logits_file.open(args.at("--logits"),std::ios::binary);if(!logits_file) throw std::runtime_error("cannot open logits output");}
    std::vector<int> forced;
    if(args.count("--force-tokens")) {forced=read_json(args.at("--force-tokens")).get<std::vector<int>>();if(chunks.size()!=1) throw std::runtime_error("teacher forcing requires one chunk");}
    int repeats=std::stoi(get("--repeat","1"));
    if(repeats<1 || repeats>100) throw std::runtime_error("repeat must be 1..100");
    for(int repeat=0;repeat<repeats;++repeat) {
    waves.clear(); report["chunks"]=json::array(); report["iteration"]=repeat+1; report["seed"]=seed; rng.seed(seed);
    if(decoder) decoder->seed(seed);
    double lm_seconds=0,decoder_seconds=0;size_t total_tokens=0;
    for(auto & chunk:chunks) {
        auto ids=kitten::prompt(tm,tokenize(vocab,chunk),ref_text,ref_audio,emotion,!args.count("--no-reference") && gen.value("use_reference_prompt",true));
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
        if(decoder) {begin=clock_type::now();waves.push_back(decoder->decode(audio_ids,ref.dump()));decoder_seconds+=std::chrono::duration<double>(clock_type::now()-begin).count();}
    }
    report["lm_seconds"]=lm_seconds;report["decoder_seconds"]=decoder_seconds;report["generated_tokens"]=total_tokens;
    if(decoder) {auto audio=kitten::join(waves,gap);write_wav(get("--output","output.wav"),audio);report["audio_seconds"]=audio.size()/24000.0;report["rtf"]=(lm_seconds+decoder_seconds)/(audio.size()/24000.0);}
    if(args.count("--report")) write_json(args.at("--report"),report);
    if(args.count("--quiet") && decoder) std::cout<<"Saved "<<get("--output","output.wav")<<'\n';
    else std::cout<<report.dump(2)<<'\n';
    }
    return 0;
} catch(const std::exception & e) {std::cerr<<"kitten-tts: "<<e.what()<<'\n';return 1;}
