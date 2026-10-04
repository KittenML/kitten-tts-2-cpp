#pragma once
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace kitten {
using json = nlohmann::ordered_json;
struct token_map {
    int base, count, speech_start, speech_end, text_start, start, stop, final_seg;
    int ref_text_start, ref_text_end, ref_speech_start, ref_speech_end;
    explicit token_map(const json & j) : base(j.at("audio_id_base")), count(j.at("num_audio_tokens")),
        speech_start(j.at("speech_start_id")), speech_end(j.at("speech_end_id")),
        text_start(j.at("text_start_id")), start(j.at("start_id")), stop(j.at("stop_id")),
        final_seg(j.value("final_seg_id", 0)), ref_text_start(j.value("reference_text_start_id",0)),
        ref_text_end(j.value("reference_text_end_id",0)), ref_speech_start(j.value("reference_speech_start_id",0)),
        ref_speech_end(j.value("reference_speech_end_id",0)) {}
};
inline std::vector<int> prompt(const token_map & t, const std::vector<int> & text,
    const std::vector<int> & ref_text, const std::vector<int> & ref_audio, const std::vector<int> & emotion, bool reference) {
    std::vector<int> p={t.start};
    auto append=[&](const std::vector<int> & v) { p.insert(p.end(),v.begin(),v.end()); };
    if(reference) {
        p.push_back(t.ref_text_start ? t.ref_text_start : t.text_start); append(ref_text);
        if(t.ref_text_start) p.push_back(t.ref_text_end);
        p.push_back(t.ref_text_start ? t.ref_speech_start : t.speech_start);
        for(int id:ref_audio) p.push_back(t.base+id);
        p.push_back(t.ref_text_start ? t.ref_speech_end : t.speech_end);
    }
    append(emotion); p.push_back(t.text_start); append(text);
    if(t.final_seg) p.push_back(t.final_seg);
    p.push_back(t.speech_start); return p;
}
struct sampling {
    float temperature=0.8f, top_p=0.8f, min_p=0, repetition=1.1f, run_penalty=1.3f;
    int top_k=50, window=50, grace=10;
};
inline void process(std::vector<float> & scores, const std::vector<int> & history, size_t prompt_size,
                    const token_map & t, const sampling & s) {
    if(!history.empty() && s.run_penalty!=0) {
        int last=history.back(), run=0;
        for(auto it=history.rbegin();it!=history.rend() && *it==last;++it) ++run;
        if(last>=t.base && last<t.base+t.count && run>=s.grace) scores.at(last)-=(run-s.grace+1)*s.run_penalty;
    }
    if(s.repetition!=1) {
        size_t begin=s.window>0 ? std::max(prompt_size,history.size()>size_t(s.window)?history.size()-s.window:0) : 0;
        std::vector<bool> seen(scores.size(),false);
        for(size_t i=begin;i<history.size();++i) {
            int id=history[i];
            if(id==t.speech_end || id==t.stop || id==t.base+4299 || seen.at(id)) continue;
            seen[id]=true; scores[id]=scores[id]<0?scores[id]*s.repetition:scores[id]/s.repetition;
        }
    }
    for(auto & score:scores) score/=s.temperature;
    const float neg=-std::numeric_limits<float>::infinity();
    if(s.top_k>0 && size_t(s.top_k)<scores.size()) {
        auto sorted=scores; std::nth_element(sorted.begin(),sorted.begin()+s.top_k-1,sorted.end(),std::greater<float>());
        float cutoff=sorted[s.top_k-1]; for(auto & x:scores) if(x<cutoff) x=neg;
    }
    if(s.top_p<1) {
        std::vector<size_t> order;
        for(size_t i=0;i<scores.size();++i) if(std::isfinite(scores[i])) order.push_back(i);
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return scores[a]<scores[b];});
        float peak=scores[order.back()]; double sum=0; for(float x:scores) sum+=std::exp(double(x-peak));
        double acc=0;
        for(size_t i=0;i+1<order.size();++i) {
            size_t id=order[i]; acc+=std::exp(double(scores[id]-peak))/sum;
            if(acc<=1.0-s.top_p) scores[id]=neg;
        }
    }
    if(s.min_p>0) {
        float cutoff=*std::max_element(scores.begin(),scores.end())+std::log(s.min_p);
        for(auto & x:scores) if(x<cutoff) x=neg;
    }
}
inline int sample(const std::vector<float> & scores, std::mt19937 & rng, bool greedy) {
    if(greedy) return std::max_element(scores.begin(),scores.end())-scores.begin();
    float peak=*std::max_element(scores.begin(),scores.end());
    std::vector<double> weights;
    std::vector<int> ids;
    for(size_t i=0;i<scores.size();++i) if(std::isfinite(scores[i])) {
        ids.push_back(i); weights.push_back(std::exp(double(scores[i]-peak)));
    }
    if(ids.empty()) throw std::runtime_error("no finite sampling candidates");
    return ids[std::discrete_distribution<size_t>(weights.begin(),weights.end())(rng)];
}
}
