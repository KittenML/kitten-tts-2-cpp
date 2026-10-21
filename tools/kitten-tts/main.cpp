#include "llama.h"
#include "text.h"
#include "sampling.h"
#include "decoder.h"
#include "assets.h"
#include "engine.h"
#include <chrono>
#include <clocale>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
using kitten::json;
using clock_type = std::chrono::steady_clock;
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
int main(int argc,char ** argv) try {
    std::setlocale(LC_ALL,"C.UTF-8");
    try { std::locale::global(std::locale("C.UTF-8")); } catch (const std::runtime_error &) {}
    std::map<std::string,std::string> args;
    const std::vector<std::string> flags={"--download-only","--offline","--no-normalize","--no-reference","--no-repack","--greedy","--tokens-only","--frontend","--help"};
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
          "  --frontend (JSON lines on stdin; text/chunking/sampling diagnostics)\n";
        return 0;
    }
    auto get=[&](const std::string & key,const std::string & fallback){auto it=args.find(key);return it==args.end()?fallback:it->second;};
    const std::vector<std::string> known={"--repo","--revision","--decoder","--cache-dir","--assets","--model","--text","--voice","--output","--data","--threads","--decoder-threads","--seed","--preset","--max-tokens","--temperature","--top-k","--top-p","--min-p","--repetition-penalty","--repetition-window","--run-penalty","--run-grace","--chunk-chars","--chunk-min-chars","--chunk-gap","--report","--logits","--force-tokens","--decode-tokens","--repeat"};
    for(auto & a:args) if(std::find(flags.begin(),flags.end(),a.first)==flags.end() && std::find(known.begin(),known.end(),a.first)==known.end()) throw std::runtime_error("unknown option "+a.first);
    if(args.count("--frontend")) {
        kitten_text_processing::Normalizer normalizer(get("--data",KITTEN_DEFAULT_DATA));
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
    if(args.count("--download-only")) { kitten::resolve_assets(args); std::cout<<"Assets ready\n"; return 0; }
    if(!args.count("--seed")) args["--seed"]=std::to_string(std::random_device{}() % 2147483647);
    kitten::engine engine(args);
    int repeats=std::stoi(get("--repeat","1"));
    if(repeats<1 || repeats>100) throw std::runtime_error("repeat must be 1..100");
    for(int i=0;i<repeats;++i) {
        auto result=engine.generate({});
        result.report["iteration"]=i+1;
        if(!args.count("--tokens-only") || args.count("--decode-tokens")) write_wav(get("--output","output.wav"),result.audio);
        if(args.count("--report")) write_json(args.at("--report"),result.report);
        std::cout<<result.report.dump(2)<<'\n';
    }
    return 0;
} catch(const std::exception & e) {std::cerr<<"kitten-tts: "<<e.what()<<'\n';return 1;}
