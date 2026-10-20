#include "engine.h"
#include "llama.h"
#include "../../src/llama-ext.h" // staging API: model tensors, devices, graph splits and memory breakdown
#include "ggml-backend.h"
#include "text.h"
#include "sampling.h"
#include "decoder.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>

namespace kitten {
using clock_type = std::chrono::steady_clock;
static double seconds_since(clock_type::time_point t) { return std::chrono::duration<double>(clock_type::now()-t).count(); }
static double mib(size_t bytes) { return bytes/1048576.0; }
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
// A weight is on the GPU when a GPU device owns its buffer and the buffer is not that device's pinned host buffer.
static bool weight_on_gpu(const ggml_tensor * t) {
    if(!t->buffer) return false;
    auto buft=ggml_backend_buffer_get_type(t->buffer);auto dev=ggml_backend_buft_get_device(buft);
    if(!dev) return false;
    auto type=ggml_backend_dev_type(dev);
    return (type==GGML_BACKEND_DEVICE_TYPE_GPU || type==GGML_BACKEND_DEVICE_TYPE_IGPU) && buft!=ggml_backend_dev_host_buffer_type(dev);
}
struct placement {
    std::string mode="cpu"; // cpu, gpu or hybrid
    int tq2_1_cpu=0;
    std::string first_tq2_1_cpu;
    std::vector<std::string> cpu_weights; // CPU-resident weights except the input token embedding
    json buffers=json::object();
};
static placement inspect_placement(const llama_model * model) {
    struct usage { int tensors=0; size_t bytes=0; std::map<std::string,int> types; };
    std::map<std::string,usage> by_buffer; placement p; bool any_gpu=false,input_seen=false;
    for(int32_t i=0;i<llama_model_n_tensors(model);++i) {
        const ggml_tensor * t=llama_model_get_tensor_by_index(model,i);
        bool gpu=weight_on_gpu(t);
        auto & u=by_buffer[t->buffer?ggml_backend_buffer_name(t->buffer):"none"];
        ++u.tensors;u.bytes+=t->buffer?ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(t->buffer),t):ggml_nbytes(t);++u.types[ggml_type_name(t->type)];
        if(t->type==GGML_TYPE_TQ2_1 && !gpu) {if(p.tq2_1_cpu==0) p.first_tq2_1_cpu=ggml_get_name(t); ++p.tq2_1_cpu;}
        any_gpu|=gpu;
        // the input layer (token_embd lookup) always stays on the CPU by design
        if(!gpu) {if(!input_seen && std::strcmp(ggml_get_name(t),"token_embd.weight")==0) input_seen=true; else p.cpu_weights.push_back(ggml_get_name(t));}
    }
    for(auto & [name,u]:by_buffer) p.buffers[name]={{"tensors",u.tensors},{"mib",mib(u.bytes)},{"types",u.types}};
    p.mode=!any_gpu?"cpu":p.cpu_weights.empty()?"gpu":"hybrid";
    return p;
}
// Device-wide used memory (total - free), sampled at stage ends. GGML CUDA pools only grow while a context lives,
// so the sample taken after generation sees the LM high-water mark.
struct gpu_memory {
    struct device { ggml_backend_dev_t dev; size_t total=0,baseline=0,peak=0,low=0; std::string peak_stage="baseline"; std::map<std::string,size_t> stages; bool used=false; };
    std::vector<device> devices;
    void init() {
        for(size_t i=0;i<ggml_backend_dev_count();++i) {
            auto dev=ggml_backend_dev_get(i);auto type=ggml_backend_dev_type(dev);
            if(type==GGML_BACKEND_DEVICE_TYPE_GPU || type==GGML_BACKEND_DEVICE_TYPE_IGPU) {device d;d.dev=dev;devices.push_back(d);}
        }
        for(auto & d:devices) {size_t free=0,total=0;ggml_backend_dev_memory(d.dev,&free,&total);d.total=total;d.baseline=d.peak=d.low=total-free;}
    }
    void sample(const std::string & stage) {
        for(auto & d:devices) {
            if(!d.used) continue;
            size_t free=0,total=0;ggml_backend_dev_memory(d.dev,&free,&total);size_t used=total-free;
            d.stages[stage]=std::max(d.stages[stage],used);d.low=std::min(d.low,used);
            if(used>d.peak) {d.peak=used;d.peak_stage=stage;}
        }
    }
    json to_json() const {
        json out=json::array();
        for(auto & d:devices) {
            if(!d.used) continue;
            json stages=json::object();for(auto & [k,v]:d.stages) stages[k]=mib(v);
            out.push_back({{"device",ggml_backend_dev_name(d.dev)},{"description",ggml_backend_dev_description(d.dev)},{"total_mib",mib(d.total)},
                {"baseline_used_mib",mib(d.baseline)},{"peak_used_mib",mib(d.peak)},{"peak_over_baseline_mib",mib(d.peak-d.baseline)},{"peak_stage",d.peak_stage},{"used_mib_by_stage",stages},
                // usage below the baseline means another process freed memory during the run
                {"other_process_changed_usage",d.low<d.baseline}});
        }
        return out;
    }
};
struct stage_times {
    double tokenize=0,context=0,prefill=0,decode=0,sampling=0,output=0;
    size_t prefill_tokens=0,decode_calls=0;
};
static const char * precision_note = "CUDA quantized matrix multiplies quantize activations to Q8_1 (CPU TQ2_1 kernels use Q8_K, or Q8_0 on the AMX path). The F16 output head uses FP16 products and partial sums on CUDA; requests for more than one logits row use cuBLAS FP16 compute (this runtime reads one row). Flash attention: CUDA decode on SM89/SM120 uses the vector kernel (FP32 accumulation); prefill, and decode on SM75/SM86 (inferred from code), use the tensor-core kernel (K*Q in FP32, softmax*V in FP16); the CPU accumulates softmax*V in FP16. With flash attention off, K*Q runs in FP32 on both; CUDA computes softmax*V with FP16 partial sums or cuBLAS FP16 compute where the CPU uses FP32 (inferred from code, not measured). Norm, RoPE and softmax sums run in a different order. Logits differ slightly from CPU builds; logits parity was measured on SM89 only.";

struct engine::impl {
    std::map<std::string, std::string> defaults;
    kitten::assets assets;
    json config, voices;
    std::unique_ptr<kitten_text_processing::Normalizer> normalizer;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<kitten::decoder> decoder;
    // S3 conditioning per voice, already on the decoder device
    std::map<std::string, std::shared_ptr<const kitten::decoder::voice>> conditioning;
    int gpu_layers=0, main_gpu=0;
    llama_flash_attn_type flash_attn=LLAMA_FLASH_ATTN_TYPE_AUTO;
    std::string flash_attn_arg="auto", decoder_device="cpu";
    json lm;                       // LM placement, devices and precision; null when no LM is loaded
    json startup=json::object();   // one-time load costs, reported with --timings
    gpu_memory gpu_mem;
    clock_type::time_point created=clock_type::now();
    int requests=0;
    const kitten::decoder::voice & voice(const std::string & name, double & seconds) {
        auto it=conditioning.find(name);
        if(it==conditioning.end()) {
            auto begin=clock_type::now();
            it=conditioning.emplace(name,decoder->prepare_voice(voices.at(name))).first;
            seconds+=seconds_since(begin);
        }
        return *it->second;
    }
};
engine::engine(const std::map<std::string, std::string> & args): p(std::make_unique<impl>()) {
    std::lock_guard<std::mutex> lock(inference_mutex);
    auto get=[&](const std::string & key,const std::string & fallback){auto it=args.find(key);return it==args.end()?fallback:it->second;};
    p->defaults=args;
    p->defaults.erase("--hf-token");
    int threads=std::stoi(get("--threads","8")), decoder_threads=std::stoi(get("--decoder-threads",std::to_string(threads)));
    if(threads<1 || decoder_threads<1) throw std::invalid_argument("thread counts must be positive");
    auto count=[&](const std::string & key,const std::string & value){
        try {size_t end=0;int n=std::stoi(value,&end);if(end==value.size() && n>=0) return n;} catch(const std::exception &) {}
        throw std::invalid_argument(key+" must be a non-negative integer"+(key=="--gpu-layers"?" or all":""));
    };
    std::string gpu_arg=get("--gpu-layers","0");
    p->gpu_layers=gpu_arg=="all"?-1:count("--gpu-layers",gpu_arg);
    p->main_gpu=count("--main-gpu",get("--main-gpu","0"));
    const std::map<std::string,llama_flash_attn_type> fa_types={{"auto",LLAMA_FLASH_ATTN_TYPE_AUTO},{"on",LLAMA_FLASH_ATTN_TYPE_ENABLED},{"off",LLAMA_FLASH_ATTN_TYPE_DISABLED}};
    p->flash_attn_arg=get("--flash-attn","auto");
    if(!fa_types.count(p->flash_attn_arg)) throw std::invalid_argument("--flash-attn must be auto, on or off");
    p->flash_attn=fa_types.at(p->flash_attn_arg);
    p->decoder_device=get("--decoder-device","cpu");
    if(p->decoder_device!="cpu" && p->decoder_device!="cuda") throw std::invalid_argument("--decoder-device must be cpu or cuda");
    const bool timings=args.count("--timings");
    auto started=clock_type::now();
    p->assets=resolve_assets(args);
    p->startup["assets_seconds"]=seconds_since(started);
    p->config=p->assets.config;
    started=clock_type::now();
    p->voices=read_json(p->assets.voices);
    p->startup["voices_load_seconds"]=seconds_since(started);
    started=clock_type::now();
    p->normalizer=std::make_unique<kitten_text_processing::Normalizer>(get("--data",KITTEN_DEFAULT_DATA));
    p->startup["normalizer_init_seconds"]=seconds_since(started);
    static std::once_flag initialized;
    std::call_once(initialized, [] { ggml_backend_load_all(); llama_backend_init(); });
    // the CUDA decoder (LibTorch cuda:0) shares device 0 with GGML's CUDA0
    const bool cuda_decoder=!p->assets.decoder.empty() && p->decoder_device=="cuda";
    if(!args.count("--decode-tokens")) {
        if(p->gpu_layers!=0 && !llama_supports_gpu_offload()) {
            if(ggml_backend_reg_by_name("CUDA")) throw std::runtime_error("--gpu-layers needs a visible CUDA GPU, but none was found (no NVIDIA driver, no device, or CUDA_VISIBLE_DEVICES hides it); use --gpu-layers 0 for CPU");
            throw std::runtime_error("--gpu-layers needs a GPU build (for example -DGGML_CUDA=ON) and a visible GPU; use --gpu-layers 0 for CPU");
        }
        if(timings && (p->gpu_layers!=0 || cuda_decoder)) p->gpu_mem.init();
        if(cuda_decoder && !p->gpu_mem.devices.empty()) p->gpu_mem.devices[0].used=true;
        auto mp=llama_model_default_params();mp.n_gpu_layers=p->gpu_layers;mp.use_extra_bufts=!args.count("--no-repack");
        // --gpu-layers 0 gives the LM no GPU device, so GPU builds do not offload prompt ops and match CPU builds.
        // Otherwise one GPU holds the whole LM; splitting a 1.7B model across GPUs only adds transfers.
        static ggml_backend_dev_t no_gpu[]={nullptr};
        if(p->gpu_layers==0) mp.devices=no_gpu;
        else {mp.split_mode=LLAMA_SPLIT_MODE_NONE;mp.main_gpu=p->main_gpu;}
        started=clock_type::now();
        p->model.reset(llama_model_load_from_file(p->assets.model.c_str(),mp));
        if(!p->model) throw std::runtime_error(p->gpu_layers!=0?"cannot load language model (the llama log above gives the cause, for example a --main-gpu index that is not below the number of GPUs)":"cannot load language model");
        p->startup["model_load_seconds"]=seconds_since(started);
        token_map tm(p->config.at("token_map"));
        if(tm.base+tm.count>llama_vocab_n_tokens(llama_model_get_vocab(p->model.get()))) throw std::runtime_error("token map exceeds vocabulary");
        started=clock_type::now();
        auto pl=inspect_placement(p->model.get());
        p->startup["placement_seconds"]=seconds_since(started);
        json devices=json::array();
        for(int i=0;i<llama_model_n_devices(p->model.get());++i) {
            auto dev=llama_model_get_device(p->model.get(),i);
            devices.push_back(std::string(ggml_backend_dev_name(dev))+" ("+ggml_backend_dev_description(dev)+")");
            for(auto & d:p->gpu_mem.devices) if(d.dev==dev) d.used=true;
        }
        if(p->gpu_layers!=0 || timings) {
            std::string summary;
            for(auto & buf:pl.buffers.items()) {
                const auto & b=buf.value();
                summary+=" "+buf.key()+": "+std::to_string(b.at("tensors").get<int>())+" weights, "+std::to_string(int(b.at("mib").get<double>()+0.5))+" MiB (";
                bool first=true;for(auto & type:b.at("types").items()) {summary+=(first?"":", ")+type.key()+" "+std::to_string(type.value().get<int>());first=false;}
                summary+=");";
            }
            std::fprintf(stderr,"kitten-tts: LM placement %s, gpu-layers %s:%s\n",pl.mode.c_str(),p->gpu_layers<0?"all":std::to_string(p->gpu_layers).c_str(),summary.c_str());
        }
        if(p->gpu_layers!=0) {
            if(pl.mode=="cpu") throw std::runtime_error("--gpu-layers was given, but no LM weight was placed on a GPU");
            if(pl.tq2_1_cpu>0 && !args.count("--allow-hybrid")) {
                throw std::runtime_error(std::to_string(pl.tq2_1_cpu)+" TQ2_1 weight tensors stay on the CPU (for example "+pl.first_tq2_1_cpu+
                    "): --gpu-layers is below 29 or the GPU backend cannot run TQ2_1. Use a build with the TQ2_1 GPU repack and --gpu-layers 99, or pass --allow-hybrid to run those matrix multiplies on the CPU");
            }
            if(pl.mode=="hybrid") {
                std::string names;for(size_t i=0;i<pl.cpu_weights.size() && i<4;++i) names+=(i?", ":"")+pl.cpu_weights[i];
                std::fprintf(stderr,"kitten-tts: hybrid: %zu LM weight tensors run on the CPU (%s%s)\n",pl.cpu_weights.size(),names.c_str(),pl.cpu_weights.size()>4?", ...":"");
            }
        }
        p->lm={{"gpu_layers",p->gpu_layers},{"placement",pl.mode},{"devices",devices},{"weights_by_buffer",pl.buffers},{"cpu_weights",pl.cpu_weights.size()},
            {"tq2_1_on_cpu",pl.tq2_1_cpu},{"flash_attn_requested",p->flash_attn_arg}};
        if(p->gpu_layers!=0) p->lm["main_gpu"]=p->main_gpu;
        if(pl.mode!="cpu") p->lm["precision"]=precision_note;
        if(timings) p->gpu_mem.sample("model_load");
    }
    started=clock_type::now();
    if(!p->assets.decoder.empty()) p->decoder=std::make_unique<kitten::decoder>(p->assets.decoder,decoder_threads,0,p->decoder_device);
    p->startup["decoder_load_seconds"]=seconds_since(started);
    if(timings) p->gpu_mem.sample("decoder_load");
    if(!p->lm.is_null()) {
        const auto mode=p->lm.at("placement").get<std::string>();
        json cpu_stages={"text normalization","tokenization","token_embd lookup (input layer)"};
        if(mode!="cpu") cpu_stages.push_back("logits copy to host (one row per step)");
        cpu_stages.push_back("sampling");
        if(p->decoder) {for(auto & s:p->decoder->cpu_stages()) cpu_stages.push_back(s); cpu_stages.push_back("chunk joins and audio output");}
        if(mode!="gpu") cpu_stages.push_back(mode=="cpu"?"speech LM":"speech LM weights not offloaded");
        p->lm["cpu_stages"]=cpu_stages;
    }
}
engine::~engine() = default;
json engine::metadata() const {
    json names=json::array(); for(auto it=p->voices.begin();it!=p->voices.end();++it) names.push_back(it.key());
    json result={{"config",p->config},{"voices",names},{"decoder_device",p->decoder_device}};
    if(!p->lm.is_null()) result["lm"]=p->lm;
    return result;
}
std::string engine::normalize(const std::string & text) {
    std::lock_guard<std::mutex> lock(inference_mutex);
    return kitten::normalize(text,*p->normalizer,true);
}
generation_result engine::generate(const std::map<std::string, std::string> & options,
    const std::function<bool(const std::vector<float> &)> & on_chunk) {
    std::lock_guard<std::mutex> lock(inference_mutex);
    const auto request_begin=clock_type::now();
    auto args=p->defaults; for(const auto & pair:options) args[pair.first]=pair.second;
    auto get=[&](const std::string & key,const std::string & fallback){auto it=args.find(key);return it==args.end()?fallback:it->second;};
    const auto & config=p->config;
    const auto & voices=p->voices;
    auto & model=p->model;
    auto & decoder=p->decoder;
    auto & normalizer=*p->normalizer;
    const bool timings=args.count("--timings");
    token_map tm(config.at("token_map"));
    std::string voice=get("--voice",config.value("default_voice","Bruno"));
    if(!voices.contains(voice)) throw std::invalid_argument("unknown voice: "+voice);
    int threads=std::stoi(get("--threads","8"));
    int seed=std::stoi(get("--seed",std::to_string(std::random_device{}() % 2147483647))), max_tokens=std::stoi(get("--max-tokens","1000"));
    if(threads<1 || max_tokens<1) throw std::invalid_argument("thread and token counts must be positive");
    double voice_prepare_seconds=0;
    if(args.count("--decode-tokens")) {
        auto tokens=read_json(args.at("--decode-tokens")).get<std::vector<int>>();
        tokens.erase(std::remove_if(tokens.begin(),tokens.end(),[&](int id){return id<0 || id>=tm.count;}),tokens.end());
        if(tokens.empty()) throw std::runtime_error("no valid codec tokens");
        if(!decoder) throw std::runtime_error("decoder is not loaded");
        decoder->seed(seed);
        const auto & conditioning=p->voice(voice,voice_prepare_seconds);
        auto begin=clock_type::now();
        auto audio=decoder->decode(tokens,conditioning);
        json report={{"samples",audio.size()},{"decoder_seconds",std::chrono::duration<double>(clock_type::now()-begin).count()}};
        return {std::move(audio),std::move(report)};
    }
    if(!model) throw std::runtime_error("language model is not loaded");
    const auto & ref=voices.at(voice);
    auto gen=config.value("generation",json::object());
    int chunk_chars=std::stoi(get("--chunk-chars",std::to_string(gen.value("chunk_chars",380))));
    int chunk_min=std::stoi(get("--chunk-min-chars",std::to_string(gen.value("chunk_min_chars",130))));
    float gap=std::stof(get("--chunk-gap","0.16"));
    if(!std::isfinite(gap) || gap<0 || gap>60) throw std::runtime_error("invalid chunk gap");
    auto started=clock_type::now();
    auto spoken=kitten::normalize(args.at("--text"),normalizer,!args.count("--no-normalize"));
    auto chunks=kitten::split(spoken,chunk_chars,chunk_min);
    const double normalize_seconds=seconds_since(started);
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
    const kitten::decoder::voice * conditioning=decoder?&p->voice(voice,voice_prepare_seconds):nullptr;
    std::vector<std::vector<float>> waves;
    json report={{"normalized",spoken},{"voice",voice},{"sample_rate",24000},{"chunks",json::array()}};
    if(!p->lm.is_null()) report["lm"]=p->lm;
    if(decoder && args.count("--decoder-device")) report["decoder_device"]=p->decoder_device;
    std::ofstream logits_file;
    if(args.count("--logits")) {logits_file.open(args.at("--logits"),std::ios::binary);if(!logits_file) throw std::runtime_error("cannot open logits output");}
    std::vector<int> forced;
    if(args.count("--force-tokens")) {forced=read_json(args.at("--force-tokens")).get<std::vector<int>>();if(chunks.size()!=1) throw std::runtime_error("teacher forcing requires one chunk");}
    report["iteration"]=1; report["seed"]=seed;
    if(decoder) decoder->seed(seed);
    json llama_buffers=json::object();
    bool splits_warned=false;
    // Splits beyond the input lookup mean an op runs on the CPU although every weight is on the GPU.
    auto check_splits=[&](llama_context * c,const char * graph){
        if(!report.contains("lm")) return;
        int n=llama_get_graph_splits(c);auto & gs=report["lm"]["graph_splits"];
        if(!gs.contains(graph) || gs[graph].get<int>()<n) gs[graph]=n;
        if(n>2 && report["lm"]["placement"]=="gpu" && !splits_warned) {
            std::fprintf(stderr,"kitten-tts: warning: the %s graph has %d backend splits, expected 2 with the whole LM on the GPU: some ops run on the CPU (GGML_SCHED_DEBUG=2 lists them)\n",graph,n);
            splits_warned=true;
        }
    };
    double lm_seconds=0,decoder_seconds=0;size_t total_tokens=0;
    stage_times st;
    for(auto & chunk:chunks) {
        auto t0=clock_type::now();
        auto ids=kitten::prompt(tm,tokenize(vocab,chunk),ref_text,ref_audio,emotion,args.count("--use-reference") ? args.at("--use-reference")=="1" : (!args.count("--no-reference") && gen.value("use_reference_prompt",true)));
        st.tokenize+=seconds_since(t0);
        int budget=chunks.size()==1?max_tokens:std::max(200,std::min(max_tokens,int(kitten::characters(chunk)/20.0*25*1.8)));
        if(args.count("--force-tokens")) budget=forced.size()+1;
        auto cp=llama_context_default_params();cp.n_ctx=ids.size()+budget;cp.n_batch=512;cp.n_ubatch=512;cp.n_threads=threads;cp.n_threads_batch=threads;cp.no_perf=false;
        // Only the last position's logits are read, so reserve one output row instead of n_batch rows.
        cp.n_outputs_max=1;cp.flash_attn_type=p->flash_attn;
        t0=clock_type::now();
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        if(!ctx) throw std::runtime_error("cannot initialize context");
        st.context+=seconds_since(t0);
        if(report.contains("lm")) report["lm"]["flash_attn"]=llama_get_flash_attn(ctx.get())?"on":"off";
        if(timings) {
            p->gpu_mem.sample("context_init");
            for(auto & entry:llama_get_memory_breakdown(ctx.get())) {
                auto & b=llama_buffers[ggml_backend_buft_name(entry.first)];
                const auto & m=entry.second;
                auto keep_max=[&](const char * key,size_t bytes){double v=mib(bytes);if(!b.contains(key) || b[key].get<double>()<v) b[key]=v;};
                keep_max("model_mib",m.model);keep_max("context_mib",m.context);keep_max("compute_mib",m.compute);
            }
        }
        auto begin=clock_type::now();
        { batch b(1,speaker.size());b.b.n_tokens=1;std::copy(speaker.begin(),speaker.end(),b.b.embd);b.position(0,0,false);
          if(llama_decode(ctx.get(),b.b)) throw std::runtime_error("speaker prefill failed"); }
        for(size_t offset=1;offset<ids.size();) {
            int n=std::min(size_t(512),ids.size()-offset);batch b(n);b.b.n_tokens=n;
            for(int i=0;i<n;++i) {b.b.token[i]=ids[offset+i];b.position(i,offset+i,offset+i+1==ids.size());}
            if(llama_decode(ctx.get(),b.b)) throw std::runtime_error("prompt prefill failed");
            offset+=n;
        }
        // GPU backends run asynchronously; wait so the prefill time includes the device work.
        llama_synchronize(ctx.get());
        double prefill=seconds_since(begin),chunk_decode=0,chunk_sampling=0;size_t chunk_calls=0;
        check_splits(ctx.get(),"prefill");
        st.prefill+=prefill;st.prefill_tokens+=ids.size();
        auto history=ids;std::vector<int> audio_ids,generated;bool ended=false;
        for(int step=0;step<budget;++step) {
            t0=clock_type::now();
            const float * raw=llama_get_logits_ith(ctx.get(),-1); // waits for the previous forward pass
            chunk_decode+=seconds_since(t0);
            if(logits_file.is_open()) logits_file.write(reinterpret_cast<const char *>(raw),n_vocab*sizeof(float));
            if(args.count("--force-tokens") && step==int(forced.size())) break;
            t0=clock_type::now();
            std::vector<float> scores(raw,raw+n_vocab);kitten::process(scores,history,ids.size(),tm,settings);
            int token=args.count("--force-tokens")?forced.at(step):kitten::sample(scores,rng,args.count("--greedy"));
            chunk_sampling+=seconds_since(t0);
            if(token<0 || token>=n_vocab) throw std::runtime_error("forced token out of range");
            generated.push_back(token);history.push_back(token);
            if(!args.count("--force-tokens") && (token==tm.speech_end || token==tm.stop)) {ended=true;break;}
            if(token>=tm.base && token<tm.base+tm.count) audio_ids.push_back(token-tm.base);
            if(step+1<budget) {
                t0=clock_type::now();
                batch b(1);b.b.n_tokens=1;b.b.token[0]=token;b.position(0,history.size()-1,true);if(llama_decode(ctx.get(),b.b)) throw std::runtime_error("generation failed");
                chunk_decode+=seconds_since(t0);++chunk_calls;
                if(chunk_calls==1) check_splits(ctx.get(),"decode");
            }
        }
        double elapsed=std::chrono::duration<double>(clock_type::now()-begin).count();lm_seconds+=elapsed;total_tokens+=generated.size();
        st.decode+=chunk_decode;st.sampling+=chunk_sampling;st.decode_calls+=chunk_calls;
        report["chunks"].push_back({{"text",chunk},{"prompt",ids},{"generated",generated},{"audio_tokens",audio_ids},{"terminated",ended},{"lm_seconds",elapsed}});
        if(timings) {
            auto & c=report["chunks"].back();
            c["prefill_seconds"]=prefill;c["prefill_tokens"]=ids.size();c["decode_seconds"]=chunk_decode;c["decode_calls"]=chunk_calls;c["sampling_seconds"]=chunk_sampling;
        }
        llama_perf_context_print(ctx.get());
        if(timings) p->gpu_mem.sample("generation");
        // Free the KV cache and compute buffers before decoding audio, to lower peak memory.
        t0=clock_type::now();ctx.reset();st.context+=seconds_since(t0);
        if(decoder) {
            begin=clock_type::now();auto wave=decoder->decode(audio_ids,*conditioning);
            double d=std::chrono::duration<double>(clock_type::now()-begin).count();decoder_seconds+=d;
            if(timings) {report["chunks"].back()["decoder_seconds"]=d;p->gpu_mem.sample("decoder");}
            if(on_chunk && !wave.empty() && !on_chunk(wave)) throw std::runtime_error("generation cancelled");
            waves.push_back(std::move(wave));
        }
    }
    report["lm_seconds"]=lm_seconds;report["decoder_seconds"]=decoder_seconds;report["generated_tokens"]=total_tokens;
    generation_result result;
    double audio_seconds=0;
    if(decoder) {
        auto t0=clock_type::now();
        result.audio=kitten::join(waves,gap);
        st.output=seconds_since(t0);audio_seconds=result.audio.size()/24000.0;
        report["audio_seconds"]=audio_seconds;report["rtf"]=(lm_seconds+decoder_seconds)/audio_seconds;
    }
    const int request=++p->requests;
    if(timings) {
        double total=seconds_since(request_begin),loop=st.decode+st.sampling;
        auto rate=[](double n,double s){return s>0?n/s:0.0;};
        json t={{"startup",p->startup},{"normalize_seconds",normalize_seconds},{"voice_prepare_seconds",voice_prepare_seconds},{"tokenize_seconds",st.tokenize},{"context_seconds",st.context},
            {"prefill_seconds",st.prefill},{"prefill_tokens",st.prefill_tokens},{"prefill_tokens_per_second",rate(st.prefill_tokens,st.prefill)},
            {"decode_seconds",st.decode},{"decode_calls",st.decode_calls},{"decode_tokens_per_second",rate(st.decode_calls,st.decode)},
            {"sampling_seconds",st.sampling},{"sampling_share",loop>0?st.sampling/loop:0.0},
            {"generated_tokens",total_tokens},{"generation_tokens_per_second",rate(total_tokens,loop)},
            {"decoder_seconds",decoder_seconds},{"output_seconds",st.output},{"total_seconds",total}};
        if(decoder) {t["audio_seconds"]=audio_seconds;t["rtf"]=(lm_seconds+decoder_seconds)/audio_seconds;t["rtf_total"]=total/audio_seconds;}
        // wall time since the engine was created (for the CLI, about the process start); the first request includes every startup cost
        t["process_seconds"]=seconds_since(p->created);
        t["llama_buffers_mib"]=llama_buffers;
        auto gm=p->gpu_mem.to_json();
        if(!gm.empty()) {t["gpu_memory"]=gm;t["gpu_memory_method"]="device-wide used memory (total minus free) sampled at stage ends; the baseline is taken after this process creates its CUDA context and before the model load; valid only while no other process uses the device";}
        report["timings"]=t;
        std::fprintf(stderr,"kitten-tts timings, request %d:\n",request);
        if(request==1) std::fprintf(stderr,"  startup         model load %.3f s, placement check %.3f s, decoder load %.3f s, normalizer init %.3f s (once)\n",
            p->startup.value("model_load_seconds",0.0),p->startup.value("placement_seconds",0.0),p->startup.value("decoder_load_seconds",0.0),p->startup.value("normalizer_init_seconds",0.0));
        std::fprintf(stderr,"  normalize       %8.3f s\n  voice prepare   %8.3f s\n",normalize_seconds,voice_prepare_seconds);
        std::fprintf(stderr,"  tokenize        %8.3f s\n  context         %8.3f s  (create and free, %zu chunks)\n",st.tokenize,st.context,chunks.size());
        std::fprintf(stderr,"  LM prefill      %8.3f s  (%zu positions, %.0f tok/s)\n",st.prefill,st.prefill_tokens,rate(st.prefill_tokens,st.prefill));
        std::fprintf(stderr,"  LM decode       %8.3f s  (%zu forward passes, %.1f tok/s)\n",st.decode,st.decode_calls,rate(st.decode_calls,st.decode));
        std::fprintf(stderr,"  sampling        %8.3f s  (%.1f%% of the generation loop; %zu tokens at %.1f tok/s overall)\n",st.sampling,loop>0?100*st.sampling/loop:0.0,total_tokens,rate(total_tokens,loop));
        std::fprintf(stderr,"  decoder         %8.3f s\n  output          %8.3f s\n  total           %8.3f s\n",decoder_seconds,st.output,total);
        if(decoder) std::fprintf(stderr,"  audio           %8.3f s  RTF %.3f (LM + decoder), %.3f (total)\n",audio_seconds,(lm_seconds+decoder_seconds)/audio_seconds,total/audio_seconds);
        std::fprintf(stderr,"  since load      %8.3f s  (since the engine was created, startup included)\n",t["process_seconds"].get<double>());
        for(auto & d:gm) std::fprintf(stderr,"  GPU memory      %s peak %.0f MiB used of %.0f (baseline %.0f, +%.0f, at %s)%s\n",d["device"].get<std::string>().c_str(),
            d["peak_used_mib"].get<double>(),d["total_mib"].get<double>(),d["baseline_used_mib"].get<double>(),d["peak_over_baseline_mib"].get<double>(),d["peak_stage"].get<std::string>().c_str(),
            d["other_process_changed_usage"].get<bool>()?"; another process changed device memory during the run, so these numbers are not reliable":"");
    }
    result.report=std::move(report);
    return result;
}
}
