#include "text.h"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cwctype>
#include <functional>
#include <locale>
#include <regex>
#include <stdexcept>

namespace kitten {
using W = std::wstring;
static W wide(const std::string & s) { return std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(s); }
static std::string utf8(const W & s) { return std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(s); }
static W trim(W s) {
    const auto ws = [](wchar_t c) { return std::iswspace(c); };
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), ws));
    s.erase(std::find_if_not(s.rbegin(), s.rend(), ws).base(), s.end());
    return s;
}
static W sub(const W & s, const W & pat, const W & repl) { return std::regex_replace(s, std::wregex(pat), repl); }
static W replace(const W & s, const std::wregex & re, const std::function<W(const std::wsmatch &)> & fn) {
    W out; size_t pos = 0;
    for (std::wsregex_iterator it(s.begin(), s.end(), re), end; it != end; ++it) {
        out += s.substr(pos, it->position() - pos); out += fn(*it); pos = it->position() + it->length();
    }
    return out + s.substr(pos);
}
static const std::wregex tags(LR"(\[(mundane|nervous|tender|angry|excited|stern|sad|contemplative|surprised|joyful)\]|<(pause|sigh|gasp|laugh|giggle|sob|scoff|growl|um|gulp)>)", std::regex::icase);
static const std::wregex emph(LR"(\(\(\(([^()\n]{1,80})\)\)\))");
bool has_expression(const std::string & s) { W w = wide(s); return std::regex_search(w,tags) || std::regex_search(w,emph); }
size_t characters(const std::string & s) { return wide(s).size(); }
static W dashes(W s) {
    s = sub(s, L"\\s*(?:[\u2014\u2013]|--)\\s*", L", ");
    s = sub(s, LR"(,\s*,+)", L", ");
    s = sub(s, LR"(\s+([,.!?;:]))", L"$1");
    s = sub(s, LR"(,\s*([.!?]))", L"$1");
    s = sub(s, LR"(\s{2,})", L" ");
    return trim(sub(s, LR"(^\s*,\s*)", L""));
}
static W terminate(W s) {
    s = sub(s, LR"(\s+([?!]))", L"$1");
    for (auto & c : s) { if (std::iswalpha(c)) { c = std::towupper(c); break; } }
    s = trim(s);
    if (s.empty() || W(L".!?\"\u2019\u201d')]").find(s.back()) != W::npos) { return s; }
    s = trim(sub(s, L"[-,;:\u2013\u2014]+\\s*$", L""));
    return s.empty() ? s : s + L".";
}
std::string normalize(const std::string & input, const kitten_text_processing::Normalizer & backend, bool enabled) {
    W s = wide(input); std::vector<W> saved;
    const auto protect = [&](const std::wsmatch & m) { saved.push_back(m.str()); return L"\ue000" + std::to_wstring(saved.size()-1) + L"\ue001"; };
    s = replace(s, tags, protect); s = replace(s, emph, protect);
    s = sub(s, LR"(\s*\n+\s*)", L" ");
    s = sub(s, LR"(([a-z])([.!?;:])(?=[A-Z]))", L"$1$2 ");
    s = sub(s, L"(?:\\.\\s+){2,}\\.|\u2026|\\.\\.\\.", L", ");
    W math;
    for (wchar_t c : s) {
        switch(c) {
#include "math.inc"
        default: math += c;
        }
    }
    s = sub(math, L"[ \\t]{2,}", L" ");
    s = replace(s, std::wregex(LR"(\b([vV])(\d+(?:\.\d+)+))"), [](const std::wsmatch & m) { return m[1].str()+L" "+sub(m[2], L"\\.", L" point "); });
    s = sub(s, LR"((\d)\s*([?!])(?=\s|$))", L"$1 $2");
    s = replace(s, std::wregex(LR"(\[([^\[\]\n]{1,400})\]|<([^<>\n]{1,60})>|\(([^()\n]{1,400})\))"), [](const std::wsmatch & m) {
        for (size_t i=1; i<m.size(); ++i) { if (m[i].matched) { return L", "+trim(m[i])+L","; } } return W();
    });
    while (true) { W next = sub(s, LR"(,\s*([,.!?;:]))", L"$1"); if (next == s) break; s = next; }
    s = dashes(sub(s,L"[\u2022\u2023\u2043\u25e6\u25b8\u21b3\u2192\u2190\u2191\u2193]+", L" "));
    if (!trim(s).empty()) {
        W spoken;
        if (enabled) {
            W quoted = replace(s, std::wregex(LR"xxx("([^"\n]*)")xxx"), [](const std::wsmatch & m) { return L"\u201c"+m[1].str()+L"\u201d"; });
            try { spoken = trim(wide(backend.normalize_text(utf8(quoted)))); } catch (const std::exception &) {}
        }
        s = terminate(spoken.empty() ? s : dashes(spoken));
    }
    for (size_t i=0; i<saved.size(); ++i) {
        W marker = L"\ue000"+std::to_wstring(i)+L"\ue001"; size_t pos;
        while ((pos=s.find(marker)) != W::npos) { s.replace(pos, marker.size(), saved[i]); }
    }
    return utf8(s);
}
std::vector<std::string> split(const std::string & text, int max_chars, int min_chars) {
    W s = trim(wide(text));
    if (s.empty()) return {};
    if (max_chars <= 0 || s.size() <= size_t(max_chars)) return {utf8(s)};
    std::vector<W> sentences, parts; W current;
    // Match the terminal character separately; Python's boundary consumes closing quotes.
    std::wregex boundary(L"[.!?][\"\u2019\u201d')\\]]*\\s+");
    size_t pos=0;
    for (std::wsregex_iterator it(s.begin(),s.end(),boundary), end; it!=end; ++it) {
        sentences.push_back(trim(s.substr(pos,it->position()+1-pos))); pos=it->position()+it->length();
    }
    if(pos<s.size()) sentences.push_back(trim(s.substr(pos)));
    for (W sentence : sentences) {
        while(sentence.size()>size_t(max_chars)) {
            size_t best=0;
            for (const W sep : {W(L", "),W(L"; "),W(L" \u2014 ")}) {
                auto found=sentence.substr(0,max_chars).rfind(sep); if(found!=W::npos) best=std::max(best,found);
            }
            size_t cut=best>size_t(max_chars/3) ? best+1 : max_chars;
            if(!current.empty()) { parts.push_back(current); current.clear(); }
            parts.push_back(trim(sentence.substr(0,cut))); sentence=trim(sentence.substr(cut));
        }
        if(current.empty()) current=sentence;
        else if(current.size()+1+sentence.size()<=size_t(max_chars)) current+=L" "+sentence;
        else { parts.push_back(current); current=sentence; }
    }
    if(!current.empty()) parts.push_back(current);
    bool changed=true;
    while(changed && parts.size()>1 && min_chars>0) {
        changed=false;
        for(size_t i=0;i<parts.size();++i) {
            if(parts[i].size()>=size_t(min_chars)) continue;
            bool prev=i>0 && parts[i-1].size()+1+parts[i].size()<=size_t(int(max_chars*1.4));
            bool next=i+1<parts.size() && parts[i+1].size()+1+parts[i].size()<=size_t(int(max_chars*1.4));
            if(prev && (!next || parts[i-1].size()<=parts[i+1].size())) { parts[i-1]+=L" "+parts[i]; parts.erase(parts.begin()+i); }
            else if(next) { parts[i]+=L" "+parts[i+1]; parts.erase(parts.begin()+i+1); }
            else continue;
            changed=true; break;
        }
    }
    std::vector<std::string> result;
    for(auto & part:parts) { if(!part.empty() && W(L".!?,;:\"\u2019\u201d')]").find(part.back())==W::npos) part+=L","; result.push_back(utf8(part)); }
    return result;
}
std::vector<float> join(const std::vector<std::vector<float>> & chunks, float gap) {
    std::vector<std::vector<float>> waves;
    for(auto & w:chunks) if(!w.empty()) waves.push_back(w);
    if(waves.empty()) throw std::runtime_error("model produced no audio tokens");
    if(waves.size()==1) return waves.front();
    std::vector<float> out;
    for(size_t i=0;i<waves.size();++i) {
        auto & w=waves[i]; size_t start=0,end=w.size();
        if(w.size()>=1200) {
            size_t first=w.size(), last=0;
            for(size_t f=0;f<w.size()/240;++f) {
                float sum=0; for(size_t j=0;j<240;++j) sum+=w[f*240+j]*w[f*240+j];
                if(std::sqrt(sum/240+1e-12f)>std::pow(10.f,-45.f/20)) {first=std::min(first,f*240);last=(f+1)*240;}
            }
            if(first!=w.size()) {start=first>480?first-480:0;end=std::min(w.size(),last+480);}
        }
        std::vector<float> part(w.begin()+start,w.begin()+end);
        const size_t fade=192;
        if(part.size()>2*fade) for(size_t j=0;j<fade;++j) {part[j]*=float(j)/(fade-1);part[part.size()-fade+j]*=1-float(j)/(fade-1);}
        out.insert(out.end(),part.begin(),part.end());
        if(i+1<waves.size()) out.insert(out.end(),size_t(24000*gap),0.f);
    }
    return out;
}
}
