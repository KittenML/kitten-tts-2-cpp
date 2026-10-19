// SPDX-License-Identifier: Apache-2.0
#include "voice_normalizer.h"

#include <unicode/rbnf.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "voice_data.hpp"
using icu::UnicodeString;
namespace kitten_voice {
namespace {
UnicodeString u(std::string_view s) {
  return UnicodeString::fromUTF8(
      icu::StringPiece(s.data(), static_cast<int32_t>(s.size())));
}
std::string utf8(const UnicodeString& s) {
  std::string out;
  s.toUTF8String(out);
  return out;
}
std::string trim(std::string s) {
  auto a = s.find_first_not_of(" \t\r\n");
  return a == std::string::npos
             ? ""
             : s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
void replace_all(std::string& s, const std::string& a, const std::string& b) {
  size_t p = 0;
  while ((p = s.find(a, p)) != std::string::npos) {
    s.replace(p, a.size(), b);
    p += b.size();
  }
}
bool digits(const std::string& s) {
  return !s.empty() && std::all_of(s.begin(), s.end(),
                                   [](char c) { return c >= '0' && c <= '9'; });
}
bool contains_digits(const std::string& s) {
  UnicodeString v = u(s);
  for (int32_t i = 0; i < v.length(); ++i)
    if (u_isdigit(v.charAt(i))) return true;
  return false;
}
std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> r;
  size_t start = 0;
  for (size_t i = 0; i <= s.size(); ++i)
    if (i == s.size() || s[i] == sep) {
      r.push_back(s.substr(start, i - start));
      start = i + 1;
    }
  return r;
}
std::string join(const std::vector<std::string>& xs,
                 const std::string& sep = " ") {
  std::string r;
  for (const auto& x : xs)
    if (!x.empty()) {
      if (!r.empty()) r += sep;
      r += x;
    }
  return r;
}
struct Number {
  std::string whole, fraction;
  char sign = 0;
};
bool grouping(const std::vector<std::string>& v) {
  if (v.empty()) return false;
  if (v.size() == 1) return digits(v[0]);
  if (!digits(v[0]) || v[0].size() > 3 || v[0].empty() || v.back().size() != 3)
    return false;
  bool western = true, indian = v[0].size() <= 2;
  for (size_t i = 1; i < v.size(); ++i) {
    if (!digits(v[i])) return false;
    if (v[i].size() != 3) western = false;
    if (i + 1 < v.size() && v[i].size() != 2) indian = false;
  }
  return western || indian;
}
std::optional<Number> parse_number(std::string s, const Profile& p) {
  s = trim(s);
  Number n;
  if (s.empty()) return {};
  if (s[0] == '-' || s[0] == '+') {
    n.sign = s[0];
    s.erase(0, 1);
  }
  if (s.empty()) return {};
  if (s.front() == '.' || s.front() == ',' || s.rfind("٫", 0) == 0) s = "0" + s;
  for (const auto& c : {std::string("\xC2\xA0"), std::string("\xE2\x80\xAF"),
                        std::string("’"), std::string("'")})
    replace_all(s, c, " ");
  replace_all(s, "٫", ".");
  replace_all(s, "٬", ",");
  size_t dot = s.rfind('.'), comma = s.rfind(',');
  char decimal = 0;
  if (dot != std::string::npos && comma != std::string::npos)
    decimal = dot > comma ? '.' : ',';
  else if (dot != std::string::npos || comma != std::string::npos) {
    char c = dot != std::string::npos ? '.' : ',';
    size_t count = std::count(s.begin(), s.end(), c);
    auto parts = split(s, c);
    if (count == 1 && c == (p.decimal_comma ? ',' : '.'))
      decimal = c;
    else if (grouping(parts))
      decimal = 0;
    else if (count == 1)
      decimal = c;
    else
      return {};
  }
  std::string integer = s;
  if (decimal) {
    size_t at = s.rfind(decimal);
    integer = s.substr(0, at);
    n.fraction = s.substr(at + 1);
    if (!digits(n.fraction)) return {};
  }
  char group = 0;
  for (char c : integer)
    if (c == '.' || c == ',' || c == ' ') {
      if (group && group != c) return {};
      group = c;
    }
  if (group) {
    auto v = split(integer, group);
    if (!grouping(v)) return {};
    n.whole = join(v, "");
  } else
    n.whole = integer;
  if (!digits(n.whole)) return {};
  return n;
}
const Profile& profile(std::string_view language) {
  std::string l(language.substr(0, language.find_first_of("-_")));
  for (char& c : l)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  for (const auto& p : profiles())
    if (p.language == l) return p;
  throw std::invalid_argument("Unsupported voice language");
}
std::string quote_regex(const std::string& s) { return "\\Q" + s + "\\E"; }
struct Context {
  const Profile& p;
  std::unique_ptr<icu::RuleBasedNumberFormat> rbnf;
  std::array<std::string, 10> digit_names;
  explicit Context(const Profile& v) : p(v) {
    UErrorCode e = U_ZERO_ERROR;
    UParseError pe{};
    rbnf = std::make_unique<icu::RuleBasedNumberFormat>(
        u(p.rules), icu::Locale(p.locale.c_str()), pe, e);
    if (U_FAILURE(e))
      throw std::runtime_error("Cannot load frozen spellout rules");
  }
  std::string spell(const std::string& value, std::string kind = "",
                    bool feminine = false) {
    if (!digits(value)) throw std::invalid_argument("Invalid integer");
    if (value.size() > 15) return digit_words(value);
    int64_t n = std::stoll(value);
    std::string rule;
    // CLDR 48's Arabic cardinal sets omit these gender distinctions.
    if (p.language == "ar" && feminine && n < 100) {
      static const char* female[] = {
          "صفر", "واحدة", "اثنتان", "ثلاث", "أربع",      "خمس",       "ست",
          "سبع", "ثمان",  "تسع",    "عشر",  "إحدى عشرة", "اثنتا عشرة"};
      if (n <= 12) return female[n];
      if (n < 20) return std::string(female[n - 10]) + " عشرة";
      if (n % 10)
        return std::string(female[n % 10]) + " و" +
               spell(std::to_string(n - n % 10));
    }
    if (kind == "year")
      rule = "%spellout-numbering-year";
    else if (kind == "ordinal")
      rule = std::find(p.rule_names.begin(), p.rule_names.end(),
                       "%spellout-ordinal") != p.rule_names.end()
                 ? "%spellout-ordinal"
                 : "%spellout-ordinal-masculine";
    else if (kind == "ordinal_feminine")
      rule = "%spellout-ordinal-feminine";
    else if (kind == "ordinal_masculine")
      rule = "%spellout-ordinal-masculine";
    else if (kind == "ordinal_feminine_locative")
      rule = "%spellout-ordinal-feminine-locative";
    else if (kind == "ordinal_neuter")
      rule = "%spellout-ordinal-neuter";
    else if (kind == "ordinal_n")
      rule = "%spellout-ordinal-n";
    else if (kind == "ordinal_genitive")
      rule = "%spellout-ordinal-masculine-genitive";
    else if (kind == "native")
      rule = "%spellout-cardinal-native-attributive";
    else if (kind == "accusative")
      rule = "%spellout-cardinal-masculine-accusative";
    else if (kind == "feminine_accusative")
      rule = "%spellout-cardinal-feminine-accusative";
    else if (feminine)
      rule = p.language == "ar" ? "%spellout-cardinal-masculine"
                                : "%spellout-cardinal-feminine";
    else if (kind == "unit")
      rule = "%spellout-cardinal-masculine";
    else if (kind.empty() &&
             (p.language == "de" || p.language == "it" || p.language == "es"))
      rule =
          "%spellout-numbering";  // eins / uno; apocopated forms require a noun
    if (rule.empty() || std::find(p.rule_names.begin(), p.rule_names.end(),
                                  rule) == p.rule_names.end()) {
      for (const auto& candidate :
           {std::string("%spellout-cardinal"),
            std::string("%spellout-cardinal-sinokorean"),
            std::string("%spellout-cardinal-masculine"),
            std::string("%spellout-numbering")})
        if (std::find(p.rule_names.begin(), p.rule_names.end(), candidate) !=
            p.rule_names.end()) {
          rule = candidate;
          break;
        }
    }
    UErrorCode e = U_ZERO_ERROR;
    UnicodeString out;
    icu::FieldPosition fp(0);
    rbnf->format(n, u(rule), out, fp, e);
    if (U_FAILURE(e)) throw std::runtime_error("Spellout failed");
    std::string result = utf8(out);
    replace_all(result, "\xC2\xAD", "");
    if (p.language == "ar") {
      replace_all(result, "ألفي", "ألفان");
      replace_all(result, "إثنان", "اثنان");
      replace_all(result, "إثنا", "اثنا");
    }
    if (contains_digits(result)) return digit_words(value);
    return result;
  }
  std::string digit_words(const std::string& v) {
    std::vector<std::string> out;
    for (char c : v)
      if (c >= '0' && c <= '9') {
        auto& word = digit_names[c - '0'];
        if (word.empty()) word = spell(std::string(1, c));
        out.push_back(word);
      }
    return join(out, p.language == "zh" || p.language == "ja" ? "" : " ");
  }
  std::string number(const Number& n, bool unit = false,
                     bool feminine = false) {
    std::string out =
        n.whole.size() > 1 && n.whole[0] == '0' && n.fraction.empty()
            ? digit_words(n.whole)
            : spell(n.whole, unit && n.fraction.empty() ? "unit" : "",
                    feminine);
    std::string gap = p.language == "zh" || p.language == "ja" ? "" : " ";
    if (!n.fraction.empty())
      out += gap + p.words[0] + gap + digit_words(n.fraction);
    if (n.sign) out = p.words[n.sign == '-' ? 1 : 2] + gap + out;
    return out;
  }
  int category(int64_t n) {
    if (p.language == "ru") {
      if (n % 10 == 1 && n % 100 != 11) return 0;
      if (n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 12 || n % 100 > 14))
        return 2;
      return 3;
    }
    if (p.language == "pl") {
      if (n == 1) return 0;
      if (n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 12 || n % 100 > 14))
        return 2;
      return 3;
    }
    if (p.language == "ar") {
      if (n == 1) return 0;
      if (n == 2) return 1;
      if (n % 100 >= 3 && n % 100 <= 10) return 2;
      if (n % 100 >= 11 && n % 100 <= 99) return 3;
      return 4;
    }
    return n == 1 ? 0 : 4;
  }
  std::string count_unit(const std::string& n,
                         const std::vector<std::string>& forms,
                         bool feminine = false) {
    int64_t v = n.size() < 16 ? std::stoll(n) : 100;
    int cat = category(v);
    std::string word = forms.size() == 5 ? forms[cat] : forms[v == 1 ? 0 : 1];
    if (p.language == "ar" && forms.size() == 5 && (v == 1 || v == 2))
      return v == 1 ? word + " " + spell(n, "unit", feminine) : word;
    return spell(n, "unit", feminine) + " " + word;
  }
  std::string money(const Number& n, const std::string& code,
                    bool accusative = false) {
    auto it = p.currencies.find(code);
    if (it == p.currencies.end()) return "";
    const Currency& cu = it->second;
    // Explicit fractional rupiah amounts retain their sen; never round them
    // away because modern display data usually omit the minor denomination.
    const int minor_digits = p.language == "id" && code == "IDR" ? 2 : cu.digits;
    bool female = code == "GBP" || code == "INR";
    std::string whole = n.whole;
    size_t nz = whole.find_first_not_of('0');
    whole = nz == std::string::npos ? "0" : whole.substr(nz);
    int64_t value = whole.size() < 16 ? std::stoll(whole) : 100;
    int cat = category(value);
    std::string major = cu.names[cat];
    if (p.language == "hi" && code == "INR")
      major = value == 1 ? "रुपया" : "रुपये";
    if (p.language == "zh" && code == "CNY") major = "元";
    if (p.language == "ja" && code == "JPY") major = "円";
    if (p.language == "ko" && code == "KRW") major = "원";
    if (p.language == "ko" && code == "USD") major = "달러";
    if (p.language == "id" && code == "IDR") major = "rupiah";
    if (p.language == "ar") {
      static const std::map<std::string, std::array<std::string, 5>> ar = {
          {"USD",
           {"دولار أمريكي", "دولاران أمريكيان", "دولارات أمريكية",
            "دولارًا أمريكيًا", "دولار أمريكي"}},
          {"EUR", {"يورو", "يوروان", "يوروهات", "يورو", "يورو"}},
          {"INR",
           {"روبية هندية", "روبيتان هنديتان", "روبيات هندية", "روبية هندية",
            "روبية هندية"}},
          {"SAR",
           {"ريال سعودي", "ريالان سعوديان", "ريالات سعودية", "ريالًا سعوديًا",
            "ريال سعودي"}},
          {"AED",
           {"درهم إماراتي", "درهمان إماراتيان", "دراهم إماراتية",
            "درهمًا إماراتيًا", "درهم إماراتي"}},
          {"OMR",
           {"ريال عماني", "ريالان عمانيان", "ريالات عمانية", "ريالًا عمانيًا",
            "ريال عماني"}},
          {"KWD",
           {"دينار كويتي", "ديناران كويتيان", "دنانير كويتية", "دينارًا كويتيًا",
            "دينار كويتي"}}};
      auto a = ar.find(code);
      if (a != ar.end()) major = a->second[cat];
    }
    bool east = p.language == "zh" || p.language == "ja";
    std::string amount = spell(whole, accusative && p.language == "ru"
                                          ? (female ? "feminine_accusative" : "accusative")
                                          : "unit", female);
    if (accusative && p.language == "ru" && value % 1000000 / 1000 == 1)
      replace_all(amount, "одну тысячу", "тысячу");
    std::string out = amount + (east ? "" : " ") + major;
    if (p.language == "ar" && value == 1) out = major + " واحد";
    if (p.language == "ar" && value == 2) out = major;
    if (!n.fraction.empty() &&
        n.fraction.find_first_not_of('0') != std::string::npos) {
      if (minor_digits <= 0 || static_cast<int>(n.fraction.size()) > minor_digits) {
        Number dec{whole, n.fraction, 0};
        out = number(dec, true, female) + (east ? "" : " ") + major;
      } else {
        std::string frac = n.fraction;
        while (static_cast<int>(frac.size()) < minor_digits) frac += '0';
        int64_t f = std::stoll(frac);
        std::string fstr = std::to_string(f);
        std::vector<std::string> forms{p.words[11], p.words[12]};
        bool ff = false;
        if (code == "GBP") {
          if (p.language == "en")
            forms = {"penny", "pence"};
          else if (p.language == "hi")
            forms = {"पेनी", "पेंस"};
        }
        if (code == "INR") {
          if (p.language == "hi")
            forms = {"पैसा", "पैसे"};
          else if (p.language == "en")
            forms = {"paisa", "paise"};
        }
        if (code == "RUB" && p.language == "ru") {
          forms = {"копейка", "копейки", "копейки", "копеек", "копейки"};
          ff = true;
        }
        if (code == "PLN" && p.language == "pl")
          forms = {"grosz", "grosza", "grosze", "groszy", "grosza"};
        if (code == "TRY" && p.language == "tr") forms = {"kuruş", "kuruş"};
        if (code == "IDR" && p.language == "id") forms = {"sen", "sen"};
        if (code == "USD" && p.language == "zh") forms = {"美分", "美分"};
        if (code == "EUR" && p.language == "es")
          forms = {"céntimo", "céntimos"};
        if (code == "EUR" && p.language == "pt")
          forms = {"cêntimo", "cêntimos"};
        if (p.language == "ru" && forms.size() == 2)
          forms = {"цент", "цента", "цента", "центов", "цента"};
        if (p.language == "pl" && forms.size() == 2)
          forms = {"cent", "centa", "centy", "centów", "centa"};
        if (p.language == "ar") {
          forms = {"سنت", "سنتان", "سنتات", "سنتًا", "سنت"};
          if (code == "SAR")
            forms = {"هللة", "هللتان", "هللات", "هللة", "هللة"};
          if (code == "OMR")
            forms = {"بيسة", "بيستان", "بيسات", "بيسة", "بيسة"};
          if (code == "AED" || code == "KWD" || code == "BHD")
            forms = {"فلس", "فلسان", "فلوس", "فلسًا", "فلس"};
        }
        std::string minor = count_unit(fstr, forms, ff);
        if (p.language == "zh" && code == "CNY") {
          minor = "";
          if (frac[0] != '0') minor += spell(std::string(1, frac[0])) + "角";
          if (frac[1] != '0') minor += spell(std::string(1, frac[1])) + "分";
          out += minor;
        } else if (east) {
          out += (p.language == "zh" ? "" : " ") + spell(fstr) +
                 forms[f == 1 ? 0 : forms.size() - 1];
        } else if (p.language == "ko" || (p.language == "id" && code == "IDR"))
          out += " " + minor;
        else
          out += " " + p.words[4] + " " + minor;
      }
    }
    if (n.sign) out = p.words[n.sign == '-' ? 1 : 2] + " " + out;
    return out;
  }
  std::string date(int y, int m, int d) {
    if (y < 1 || m < 1 || m > 12 || d < 1) return "";
    int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)) days[1] = 29;
    if (d > days[m - 1]) return "";
    std::string day = std::to_string(d), year = std::to_string(y),
                month = p.months[m - 1];
    const auto& l = p.language;
    if (l == "zh")
      return digit_words(year) + "年" + spell(std::to_string(m)) + "月" +
             spell(day) + "日";
    if (l == "ja")
      return spell(year) + "年" + spell(std::to_string(m)) + "月" + spell(day) +
             "日";
    if (l == "ko")
      return spell(year) + "년 " +
             (m == 6    ? "유"
              : m == 10 ? "시"
                        : spell(std::to_string(m))) +
             "월 " + spell(day) + "일";
    if (l == "en")
      return month + " " + spell(day, "ordinal") + ", " + spell(year, "year");
    if (l == "es" || l == "pt")
      return spell(day) + " de " + month + " de " + spell(year);
    if (l == "de")
      return spell(day, "ordinal_n") + " " + month + " " + spell(year, "year");
    if (l == "ru")
      return spell(day, "ordinal_neuter") + " " + month + " " +
             spell(year, "ordinal_genitive") + " года";
    if (l == "pl")
      return spell(day, "ordinal_genitive") + " " + month + " " +
             spell(year, "ordinal_genitive") + " roku";
    if (l == "fr")
      return (d == 1 ? std::string("premier") : spell(day)) + " " + month +
             " " + spell(year);
    if (l == "ar") {
      std::string yword = spell(year);
      replace_all(yword, "ألفان", "ألفين");
      replace_all(yword, "مائتان", "مائتين");
      for (const auto& tens : {"عشرون", "ثلاثون", "أربعون", "خمسون", "ستون",
                               "سبعون", "ثمانون", "تسعون"}) {
        std::string t = tens;
        auto at = yword.rfind(t);
        if (at != std::string::npos && at + t.size() == yword.size())
          yword.replace(at, t.size(), t.substr(0, t.size() - 4) + "ين");
      }
      return spell(day, "ordinal_masculine") + " من " + month + " عام " + yword;
    }
    return spell(day) + " " + month + " " + spell(year);
  }
  std::string clock(int h, int m, int sec, bool has_seconds,
                    const std::string& ampm, bool locative = false) {
    if (h < 0 || h > 23 || m < 0 || m > 59 || sec < 0 || sec > 59 ||
        (!ampm.empty() && (h < 1 || h > 12)))
      return "";
    int hour = h;
    if (!ampm.empty()) {
      hour = h % 12;
      if (ampm == "pm") hour += 12;
    }
    const auto& l = p.language;
    std::string hs = std::to_string(hour), ms = std::to_string(m),
                ss = std::to_string(sec);
    if (l == "en") {
      std::string o = spell(std::to_string(h));
      if (m == 0)
        o += " o'clock";
      else
        o += " " + (m < 10 ? std::string("oh ") : "") + spell(ms);
      if (has_seconds) o += " and " + count_unit(ss, {"second", "seconds"});
      if (!ampm.empty()) o += std::string(" ") + (ampm == "am" ? "a m" : "p m");
      return o;
    }
    if (l == "zh")
      return spell(hs) + "点" + spell(ms) + "分" +
             (has_seconds ? spell(ss) + "秒" : "");
    if (l == "ja")
      return spell(hs) + "時" + spell(ms) + "分" +
             (has_seconds ? spell(ss) + "秒" : "");
    if (l == "ko")
      return (ampm.empty() ? spell(hs)
                          : (ampm == "am" ? "오전 " : "오후 ") +
                                spell(std::to_string(h), "native")) + " 시 " +
             spell(ms) + " 분" + (has_seconds ? " " + spell(ss) + " 초" : "");
    std::string out;
    if (l == "hi")
      out = spell(hs) + " बजकर " + spell(ms) + " मिनट";
    else if (l == "de")
      out = spell(hs, "unit") + " Uhr " +
            (has_seconds ? count_unit(ms, {"Minute", "Minuten"}, true) : spell(ms));
    else if (l == "nl")
      out = spell(hs) + " uur " +
            (has_seconds ? count_unit(ms, {"minuut", "minuten"}) : spell(ms));
    else if (l == "fr")
      out = spell(hs, "unit", true) + " " + (hour == 1 ? "heure" : "heures") +
            " " + (has_seconds ? count_unit(ms, {"minute", "minutes"}, true) : spell(ms));
    else if (l == "it")
      out = (hour == 1 ? std::string("una") : spell(hs)) +
            (m == 0 ? " in punto" : " e " + spell(ms));
    else if (l == "es")
      out = has_seconds
                ? spell(hs, "unit", true) + (hour == 1 ? " hora, " : " horas, ") +
                      count_unit(ms, {"minuto", "minutos"})
                : (hour == 1 ? std::string("una") : spell(hs)) +
                      (m == 0 ? " en punto" : " y " + spell(ms));
    else if (l == "tr")
      out = spell(hs) + " " + (m < 10 ? spell("0") + " " : "") + spell(ms);
    else if (l == "id")
      out = spell(hs) + " " + (m < 10 ? spell("0") + " " : "") + spell(ms);
    else if (l == "pt")
      out = spell(hs, "unit", true) + " " + (hour == 1 ? "hora" : "horas") +
            " e " + spell(ms) + " " + (m == 1 ? "minuto" : "minutos");
    else if (l == "ru")
      out = count_unit(hs, {"час", "часа", "часа", "часов", "часа"}) + " " +
            count_unit(ms, {"минута", "минуты", "минуты", "минут", "минуты"},
                       true);
    else if (l == "pl")
      out = spell(hs,
                  locative ? "ordinal_feminine_locative" : "ordinal_feminine") +
            " " + (m < 10 ? spell("0") + " " : "") + spell(ms);
    else if (l == "ar") {
      static const char* hours[] = {
          "الصفر",   "الواحدة",      "الثانية",     "الثالثة", "الرابعة",
          "الخامسة", "السادسة",      "السابعة",     "الثامنة", "التاسعة",
          "العاشرة", "الحادية عشرة", "الثانية عشرة"};
      if (!ampm.empty()) {
        out = hours[h];
        if (!has_seconds && m == 30)
          out += " والنصف";
        else if (!has_seconds && m == 15)
          out += " والربع";
        else if (m != 0 || has_seconds)
          out += " و" + count_unit(ms, {"دقيقة", "دقيقتان", "دقائق", "دقيقة", "دقيقة"}, true);
        if (has_seconds)
          out += " و" + count_unit(ss, {"ثانية", "ثانيتان", "ثوان", "ثانية", "ثانية"}, true);
        return out + " " + (ampm == "am" ? (h == 12 ? "ليلًا" : "صباحًا")
                                         : (h == 12 ? "ظهرًا" : "مساءً"));
      }
      if (hour <= 12)
        out = hours[hour];
      else if (hour < 20)
        out = std::string(hours[hour - 10]) + " عشرة";
      else if (hour == 20)
        out = locative ? "العشرين" : "العشرون";
      else
        out = std::string(hour == 21 ? "الحادية" : hours[hour - 20]) + " و" +
              (locative ? "العشرين" : "العشرون");
      out +=
          " و" +
          count_unit(ms, {"دقيقة", "دقيقتان", "دقائق", "دقيقة", "دقيقة"}, true);
      if (locative) replace_all(out, "ون دقيقة", "ين دقيقة");
    } else
      out = spell(hs) + " " + p.words[5] + " " + spell(ms) + " " + p.words[7];
    if (has_seconds)
      out += " " + p.words[4] + " " +
             count_unit(ss, {p.words[9], p.words[10]}, l == "fr" || l == "de");
    return out;
  }
};
struct Pool {
  std::vector<std::unique_ptr<Context>> contexts;
  std::vector<Context*> idle;
  std::mutex mu;
  std::condition_variable cv;
  explicit Pool(const Profile& p) {
    for (int i = 0; i < 4; ++i) {
      contexts.push_back(std::make_unique<Context>(p));
      idle.push_back(contexts.back().get());
    }
  }
  Context* take() {
    std::unique_lock<std::mutex> l(mu);
    cv.wait(l, [&] { return !idle.empty(); });
    auto* c = idle.back();
    idle.pop_back();
    return c;
  }
  void give(Context* c) {
    {
      std::lock_guard<std::mutex> l(mu);
      idle.push_back(c);
    }
    cv.notify_one();
  }
};
Pool& pool(const Profile& p) {
  static std::mutex m;
  static std::map<std::string, std::unique_ptr<Pool>> pools;
  std::lock_guard<std::mutex> l(m);
  auto& ptr = pools[p.language];
  if (!ptr) ptr = std::make_unique<Pool>(p);
  return *ptr;
}
struct Lease {
  Pool& p;
  Context* c;
  explicit Lease(Pool& p) : p(p), c(p.take()) {}
  ~Lease() { p.give(c); }
};
std::unique_ptr<icu::RegexPattern> pattern(const std::string& s,
                                           uint32_t flags = 0) {
  UErrorCode e = U_ZERO_ERROR;
  auto* p = icu::RegexPattern::compile(u(s), flags, e);
  if (U_FAILURE(e)) throw std::runtime_error("Invalid voice grammar pattern");
  return std::unique_ptr<icu::RegexPattern>(p);
}
struct Match {
  int32_t begin, end;
  std::string output;
};
const std::string num =
    R"([+-]?(?:[0-9]++(?:[.,٬٫'’\x{00A0}\x{202F} ][0-9]++)*+|[.,٫][0-9]++))";
const std::map<std::string, int>& magnitudes() {
  static const std::map<std::string, int> values = {
      {"k", 3},         {"K", 3},          {"M", 6},         {"B", 9},
      {"thousand", 3},  {"million", 6},    {"billion", 9},   {"हज़ार", 3},
      {"हजार", 3},      {"लाख", 5},        {"करोड़", 7},      {"ألف", 3},
      {"مليون", 6},     {"مليار", 9},      {"万", 4},        {"亿", 8},
      {"億", 8},        {"만", 4},         {"억", 8},        {"천", 3},
      {"백만", 6},      {"ribu", 3},       {"juta", 6},      {"miliar", 9},
      {"bin", 3},       {"milyon", 6},     {"milyar", 9},    {"mil", 3},
      {"milhão", 6},    {"milhões", 6},    {"bilhão", 9},    {"bilhões", 9},
      {"millón", 6},    {"millones", 6},   {"milioni", 6},   {"milione", 6},
      {"miliardi", 9},  {"miliardo", 9},   {"mille", 3},     {"millions", 6},
      {"milliard", 9},  {"milliards", 9},  {"Million", 6},   {"Millionen", 6},
      {"Milliarde", 9}, {"Milliarden", 9}, {"Tausend", 3},   {"duizend", 3},
      {"miljoen", 6},   {"miljard", 9},    {"тысяч", 3},     {"тысячи", 3},
      {"миллион", 6},   {"миллиона", 6},   {"миллионов", 6}, {"миллиард", 9},
      {"миллиарда", 9}, {"миллиардов", 9}, {"tysiąc", 3},    {"tysięcy", 3},
      {"milion", 6},    {"miliony", 6},    {"milionów", 6},  {"miliard", 9},
      {"miliardy", 9},  {"miliardów", 9}};
  return values;
}
Number scale_money(Number value, const std::string& magnitude) {
  if (magnitude.empty()) return value;
  int shift = magnitudes().at(magnitude);
  if (static_cast<int>(value.fraction.size()) <= shift) {
    value.whole +=
        value.fraction + std::string(shift - value.fraction.size(), '0');
    value.fraction.clear();
  } else {
    value.whole += value.fraction.substr(0, shift);
    value.fraction.erase(0, shift);
  }
  return value;
}
std::string range_word(const Profile& p) {
  static const std::map<std::string, std::string> words = {
      {"en", "to"}, {"ar", "إلى"},  {"zh", "到"},   {"nl", "tot"},
      {"fr", "à"},  {"de", "bis"},  {"hi", "से"},    {"id", "sampai"},
      {"it", "a"},  {"ja", "から"}, {"ko", "부터"}, {"pl", "do"},
      {"pt", "a"},  {"ru", "до"},   {"es", "a"},    {"tr", "ile"}};
  return words.at(p.language);
}
const std::string codes =
    R"((?:USD|EUR|GBP|INR|BRL|RUB|KRW|JPY|CNY|TRY|PLN|CHF|SAR|AED|OMR|KWD|BHD|CAD|AUD|IDR|Rp|R\$|US\$|CA\$|AU\$|\$|€|£|₹|₽|₩|¥|￥|₺|zł|ر\.س\.?|د\.إ\.?))";
std::string currency_code(std::string c, const Profile& p) {
  static const std::map<std::string, std::string> map = {
      {"Rp", "IDR"},  {"$", "USD"},   {"US$", "USD"},  {"CA$", "CAD"},
      {"AU$", "AUD"}, {"€", "EUR"},   {"£", "GBP"},    {"₹", "INR"},
      {"R$", "BRL"},  {"₽", "RUB"},   {"₩", "KRW"},    {"₺", "TRY"},
      {"zł", "PLN"},  {"ر.س", "SAR"}, {"ر.س.", "SAR"}, {"د.إ", "AED"},
      {"د.إ.", "AED"}};
  if (c == "¥" || c == "￥") return p.language == "zh" ? "CNY" : "JPY";
  auto i = map.find(c);
  return i != map.end() ? i->second : c;
}
const std::vector<std::string>& contact_tlds() {
  static const std::vector<std::string> tlds = {
      "com", "org", "net", "io", "ai", "co", "gov", "edu", "app", "dev", "info", "biz", "me", "name", "tv", "xyz", "online", "site", "tech", "store", "example", "test",
      "us", "uk", "cn", "nl", "fr", "de", "in", "id", "it", "jp", "kr", "pl", "br", "pt", "ru", "es", "tr", "sa", "ae", "om", "au", "ca", "nz", "ch", "at", "be", "se", "no", "dk", "fi", "za", "sg", "hk", "рф", "中国", "भारत", "한국", "السعودية"};
  return tlds;
}
struct Patterns {
  std::unique_ptr<icu::RegexPattern> protect, date, east_date, money_prefix,
      money_suffix, time, dotted_time, timezone, phone, phone_extension, digit_code, percent,
      number, hour_ampm, ordinal, percent_prefix, clock_range, number_range,
      money_range_prefix, money_range_suffix;
  std::map<std::string, std::unique_ptr<icu::RegexPattern>> named_dates;
  std::map<std::string, int> month_numbers;
  Patterns() {
    std::string magnitude;
    // Longest alternatives first (e.g. millions before million).
    std::vector<std::string> names;
    for (const auto& entry : magnitudes()) names.push_back(entry.first);
    std::sort(names.begin(), names.end(),
              [](const auto& a, const auto& b) { return a.size() > b.size(); });
    for (const auto& name : names) {
      if (!magnitude.empty()) magnitude += '|';
      magnitude += quote_regex(name);
    }
    std::string tlds;
    for (const auto& suffix : contact_tlds()) {
      if (!tlds.empty()) tlds += '|';
      tlds += quote_regex(suffix);
    }
    // Numeric hostnames require a recognized suffix, so numeric/identifier
    // mixtures do not acquire a different boundary on the next pass.
    const std::string numeric_domain =
        R"((?<![\p{L}0-9_])[0-9][\p{L}0-9\-]*(?:\.[\p{L}0-9\-]+)*\.(?i:)" +
        tlds + R"()(?![\p{L}0-9_\-])(?:\:[0-9]{1,5})?(?:[/?#][^\s<>]*)?)";
    // Serving frontends replace documented controls with these opaque
    // markers. Their numeric indexes are identifiers, never spoken values.
    protect = pattern(R"(\x{E000}[0-9]+\x{E001}|)" + numeric_domain + "|" +
        R"(\b[0-9]+(?:[.,][0-9]+)*[A-Za-z_]+[0-9][A-Za-z0-9_.\-]*\b|)" +
        R"((?i:https?://|www\.)[^\s<>]+|[\p{L}0-9._%+\-]+@[\p{L}0-9.\-]+\.[\p{L}]{2,}|[A-Za-z]:\\[^\s<>]+|(?:[\p{L}][\p{L}0-9\-]*\.)+[\p{L}]{2,}(?::[0-9]{1,5})?(?:[/?#][^\s<>]*)?|(?i:(?:python|version|build|release|iOS|CUDA|HTTP|API|RFC)\s*[-/]?\s*v?[0-9][A-Za-z0-9._/\-]*)|\bv[0-9]+(?:\.[0-9]+)+(?:[-+][A-Za-z0-9.]+)?|(?<![0-9])[0-9]{1,3}(?:\.[0-9]{1,3}){2,3}(?![0-9])|(?<![0-9/])[0-9]+/[0-9]+(?![/0-9])|\b(?!(?:USD|EUR|GBP|INR|BRL|RUB|KRW|JPY|CNY|TRY|PLN|CHF|SAR|AED|OMR|KWD|BHD|CAD|AUD|IDR|Rp)[0-9])[A-Za-z_]+[0-9][A-Za-z0-9_.-]*\b|\b[0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+\b|(?<![0-9])[0-9]+[.,]{2,}[0-9.,]*|(?<![0-9/])[0-9]{1,2}/[0-9]{1,2}/[0-9]{2}(?![0-9/]))");
    date = pattern(
        R"((?<![0-9])([0-9]{4})[-/.]([0-9]{1,2})[-/.]([0-9]{1,2})(?![0-9])|(?<![0-9])([0-9]{1,2})[/.-]([0-9]{1,2})[/.-]([0-9]{4})(?![0-9]))");
    east_date = pattern(
        R"(([0-9]{4})\s*[年년]\s*([0-9]{1,2})\s*[月월]\s*([0-9]{1,2})\s*[日일])");
    money_prefix =
        pattern("(?<![A-Za-z0-9_])([+-]?)(" + codes + ")\\s*(" + num +
                ")(?:\\s*(" + magnitude + "))?(?![A-Za-z0-9_])");
    money_suffix =
        pattern("(?<![A-Za-z0-9_])(" + num + ")(?:\\s*(" + magnitude +
                "))?\\s*(" + codes + ")(?![A-Za-z0-9_])");
    money_range_prefix = pattern("(?<![A-Za-z0-9_])(" + codes + ")\\s*(" + num +
                                 ")\\s*[-–—]\\s*(?:(" + codes + "))?\\s*(" +
                                 num + ")(?![A-Za-z0-9_])");
    money_range_suffix =
        pattern("(?<![A-Za-z0-9_])(" + num + ")\\s*[-–—]\\s*(" + num +
                ")\\s*(" + codes + ")(?![A-Za-z0-9_])");
    clock_range = pattern(
        R"((?<![0-9])([0-9]{1,2}):([0-9]{2})\s*[-–—]\s*([0-9]{1,2}):([0-9]{2})(?![0-9]))");
    number_range = pattern("(?<![A-Za-z0-9_])(" + num + ")\\s*[-–—]\\s*(" +
                           num + ")(?![A-Za-z0-9_])");
    time = pattern(
        R"((?<![0-9])([0-9]{1,2}):([0-9]{2})(?::([0-9]{2}))?(?:\s*([AaPp])\.?[Mm]\.?)?(\s+बजे)?(?![0-9]))");
    hour_ampm = pattern(
        R"((?<![0-9:])([0-9]{1,2})\s*([AaPp])\.?[Mm]\.?(?![\p{L}0-9]))");
    dotted_time = pattern(
        R"((?i:(?:pukul|jam|om|kl|ore|alle|saat|الساعة)\s+)([0-9]{1,2})\.([0-9]{2})(?![0-9.]))");
    timezone =
        pattern(R"((?:UTC|GMT)\s*([+-])([0-9]{1,2}):([0-9]{2})(?![0-9]))");
    const std::string phone_body =
        R"((?:[0-9]{3}[.][0-9]{3}-[0-9]{4}|[0-9]{1,3}-[0-9]{3}-[0-9]{3}-[0-9]{4}|[0-9]{2,3} +[0-9]{3,5}-[0-9]{3,5}|\+[0-9]{7,15}|\+[0-9]{1,3}[ \-](?:\([0-9]{2,4}\)[ \-]?|[0-9])[0-9 ()\-]{4,}[0-9]|\([0-9]{2,4}\)[ \-]?[0-9][0-9 \-]{4,}[0-9]|[0-9]{2,5}-[0-9]{3,5}-[0-9]{2,5}|[0-9]{2}-[0-9]{5}|[0-9]{3}-[0-9]{4,5}|[0-9]{4,5}-[0-9]{3,5}|[0-9]{3,4} +[0-9]{3,4} +[0-9]{4}|[0-9]{5} +[0-9]{5}|[0-9]{2}(?: +[0-9]{2}){4}))";
    phone = pattern("(?<![\\p{L}0-9])(" + phone_body + ")(?![0-9])");
    phone_extension = pattern("(?<![\\p{L}0-9])(" + phone_body +
        R"(|[0-9]{7,15})\s*(?i:x|ext(?:ension)?\.?|poste|Durchwahl|toestel|interno|wew\.?|ramal|anexo|dahili|доб\.?|分机|内線|내선|إضافي|एक्सटेंशन|ekstensi)\s*[:：.]?\s*([0-9]{1,8})(?![0-9]))");
    digit_code = pattern(
        R"((?i:(?:OTP|PIN|code|phone|telephone|mobile|tel|telefoon|mobiel|téléphone|portable|tél|Handy|telepon|ponsel|telp|telefono|cellulare|telefone|celular|teléfono|móvil|telefon|cep telefonu|телефон|мобильный|код|कोड|फोन|फ़ोन|मोबाइल|رمز|هاتف|الهاتف|جوال|kode|código|コード|コード番号|電話|携帯|전화|휴대폰|코드|电话|手机|代码|编号)\s*[:：]?\s*)([0-9](?:[0-9 \-]*[0-9])?)(?![0-9]))");
    percent = pattern("(?<![A-Za-z0-9_])(" + num + ")\\s*[%％٪](?![0-9])");
    percent_prefix = pattern("[%％٪]\\s*(" + num + ")(?![0-9])");
    ordinal =
        pattern(R"((?<![\p{L}0-9_])([0-9]+)(?:st|nd|rd|th)(?![\p{L}0-9_]))",
                UREGEX_CASE_INSENSITIVE);
    number = pattern("(?<![A-Za-z0-9_])(" + num + ")(?![A-Za-z0-9_])");
    for (const auto& profile : profiles()) {
      std::string alternatives;
      for (const auto* source : {&profile.months, &profile.months_short})
        for (size_t i = 0; i < source->size(); ++i) {
          std::string name = (*source)[i];
          if (name.empty()) continue;
          UnicodeString lower = u(name);
          lower.foldCase();
          month_numbers[utf8(lower)] = static_cast<int>(i) + 1;
          if (!alternatives.empty()) alternatives += '|';
          alternatives += quote_regex(name);
        }
      named_dates[profile.language] = pattern(
          "(?<![\\p{L}0-9])(?:([0-9]{1,2})(?:st|nd|rd|th)?\\.?\\s+(?:de\\s+)?"
          "(" +
              alternatives + ")\\.?(?:\\s+de)?[ ,]+([0-9]{4})|(" +
              alternatives +
              ")\\.?\\s+([0-9]{1,2})(?:st|nd|rd|th)?[ ,]+([0-9]{4}))(?![0-9])",
          UREGEX_CASE_INSENSITIVE);
    }
  }
};
Patterns& patterns() {
  static Patterns p;
  return p;
}
struct Signals {
  bool letters = false, currency = false, percent = false, dash = false, space = false,
       colon = false, dot = false, slash = false, phone = false, control_marker = false;
  explicit Signals(const UnicodeString& text) {
    for (int32_t i = 0; i < text.length(); ++i) {
      auto ch = text.charAt(i);
      control_marker |= ch == 0xe000;
      letters |= u_isalpha(ch);
      space |= u_isUWhiteSpace(ch);
      percent |= ch == '%' || ch == 0xff05 || ch == 0x066a;
      dash |= ch == '-' || ch == 0x2013 || ch == 0x2014;
      colon |= ch == ':';
      dot |= ch == '.' || ch == ',';
      slash |= ch == '/';
      phone |= ch == '+' || ch == '(' || ch == '-';
    }
    auto raw = utf8(text);
    for (const auto& name :
         {"USD", "EUR", "GBP", "INR", "BRL", "RUB", "KRW", "JPY", "CNY",
          "TRY", "PLN", "CHF", "SAR", "AED", "OMR", "KWD", "BHD", "CAD",
          "AUD", "IDR", "Rp",  "$",   "€",   "£",   "₹",   "₽",   "₩",
          "¥",   "￥",  "₺",   "zł",  "ر.س", "د.إ"})
      if (raw.find(name) != std::string::npos) {
        currency = true;
        break;
      }
  }
};
bool may_match(const icu::RegexPattern* pattern, const Signals& s) {
  auto& p = patterns();
  if (pattern == p.protect.get()) return s.control_marker || s.letters || s.slash || s.dot;
  if (pattern == p.date.get()) return s.dash || s.dot || s.slash;
  if (pattern == p.money_prefix.get() || pattern == p.money_suffix.get())
    return s.currency;
  if (pattern == p.money_range_prefix.get() ||
      pattern == p.money_range_suffix.get())
    return s.currency && s.dash;
  if (pattern == p.percent.get() || pattern == p.percent_prefix.get())
    return s.percent;
  if (pattern == p.clock_range.get()) return s.colon && s.dash;
  if (pattern == p.number_range.get()) return s.dash;
  if (pattern == p.time.get() || pattern == p.timezone.get()) return s.colon;
  if (pattern == p.phone.get()) return s.phone || s.space;
  if (pattern == p.dotted_time.get()) return s.dot && s.letters;
  if (pattern == p.number.get()) return true;
  return s.letters;
}
// Contact punctuation is lexical glue; labels retain the caller's spelling.
// No DNS, language detector, or model call is involved.
const std::array<std::string, 12>& contact_words(const std::string& language) {
  static const std::map<std::string, std::array<std::string, 12>> words = {
    {"en", {"at", "dot", "underscore", "hyphen", "slash", "colon", "question mark", "equals", "ampersand", "hash", "percent", "extension"}},
    {"ar", {"آت", "نقطة", "شرطة سفلية", "شرطة", "شرطة مائلة", "نقطتان", "علامة استفهام", "يساوي", "علامة العطف", "علامة المربع", "بالمئة", "تحويلة"}},
    {"zh", {"艾特", "点", "下划线", "连字符", "斜杠", "冒号", "问号", "等于", "与符号", "井号", "百分号", "分机"}},
    {"nl", {"apenstaartje", "punt", "laag streepje", "streepje", "schuine streep", "dubbele punt", "vraagteken", "is gelijk aan", "ampersand", "hekje", "procent", "toestel"}},
    {"fr", {"arobase", "point", "tiret bas", "tiret", "barre oblique", "deux-points", "point d'interrogation", "égal", "esperluette", "dièse", "pour cent", "poste"}},
    {"de", {"ät", "Punkt", "Unterstrich", "Bindestrich", "Schrägstrich", "Doppelpunkt", "Fragezeichen", "gleich", "Und-Zeichen", "Raute", "Prozent", "Durchwahl"}},
    {"hi", {"ऐट", "डॉट", "अंडरस्कोर", "हाइफ़न", "स्लैश", "कोलन", "प्रश्न चिह्न", "बराबर", "ऐम्परसैंड", "हैश", "प्रतिशत", "एक्सटेंशन"}},
    {"id", {"at", "titik", "garis bawah", "tanda hubung", "garis miring", "titik dua", "tanda tanya", "sama dengan", "ampersand", "pagar", "persen", "ekstensi"}},
    {"it", {"chiocciola", "punto", "trattino basso", "trattino", "barra", "due punti", "punto interrogativo", "uguale", "e commerciale", "cancelletto", "per cento", "interno"}},
    {"ja", {"アットマーク", "ドット", "アンダースコア", "ハイフン", "スラッシュ", "コロン", "疑問符", "イコール", "アンパサンド", "シャープ", "パーセント", "内線"}},
    {"ko", {"골뱅이", "점", "밑줄", "하이픈", "슬래시", "콜론", "물음표", "등호", "앰퍼샌드", "샵", "퍼센트", "내선"}},
    {"pl", {"małpa", "kropka", "podkreślenie", "łącznik", "ukośnik", "dwukropek", "znak zapytania", "równa się", "ampersand", "kratka", "procent", "wewnętrzny"}},
    {"pt", {"arroba", "ponto", "sublinhado", "hífen", "barra", "dois-pontos", "ponto de interrogação", "igual", "e comercial", "cerquilha", "por cento", "ramal"}},
    {"ru", {"собака", "точка", "нижнее подчёркивание", "дефис", "слеш", "двоеточие", "знак вопроса", "равно", "амперсанд", "решётка", "процент", "добавочный"}},
    {"es", {"arroba", "punto", "guion bajo", "guion", "barra", "dos puntos", "signo de interrogación", "igual", "ampersand", "almohadilla", "por ciento", "extensión"}},
    {"tr", {"et", "nokta", "alt çizgi", "kısa çizgi", "eğik çizgi", "iki nokta", "soru işareti", "eşittir", "ve işareti", "kare", "yüzde", "dahili"}},
  };
  return words.at(language);
}
std::string contact_plus(const Context& c) {
  if (c.p.language == "hi") return "प्लस";
  if (c.p.language == "ar") return "زائد";
  if (c.p.language == "zh") return "加";
  return c.p.words[2];
}
bool electronic_shape(std::string_view raw) {
  if (raw.find('@') != raw.npos || raw.find("://") != raw.npos) return true;
  for (size_t i = 1; i + 1 < raw.size(); ++i)
    if (raw[i] == '.' && static_cast<unsigned char>(raw[i - 1]) > 32 &&
        static_cast<unsigned char>(raw[i + 1]) > 32 &&
        ((raw[i + 1] >= 'A' && raw[i + 1] <= 'Z') ||
         (raw[i + 1] >= 'a' && raw[i + 1] <= 'z') ||
         static_cast<unsigned char>(raw[i + 1]) >= 128)) return true;
  return false;
}
std::string electronic(std::string raw, Context& c) {
  for (unsigned char ch : raw) if (ch < 32 || ch == 127) return "";
  std::string ending;
  int parentheses = static_cast<int>(std::count(raw.begin(), raw.end(), ')')) -
                    static_cast<int>(std::count(raw.begin(), raw.end(), '('));
  int brackets = static_cast<int>(std::count(raw.begin(), raw.end(), ']')) -
                 static_cast<int>(std::count(raw.begin(), raw.end(), '['));
  while (!raw.empty()) {
    char last = raw.back();
    bool punctuation = std::string(".,!?;:\"'").find(last) != std::string::npos;
    bool closing = (last == ')' && parentheses > 0) || (last == ']' && brackets > 0);
    if (!punctuation && !closing) break;
    ending.push_back(last);
    if (last == ')') --parentheses;
    if (last == ']') --brackets;
    raw.pop_back();
  }
  std::reverse(ending.begin(), ending.end());
  auto folded = u(raw);
  folded.foldCase();
  const auto lower = utf8(folded);
  const bool scheme = lower.rfind("http://", 0) == 0 || lower.rfind("https://", 0) == 0;
  const bool email = raw.find('@') != std::string::npos && raw.find_first_of("/:\\ \t") == std::string::npos;
  if (raw.find('@') != std::string::npos && !email && !scheme) return "";
  if (!scheme && !email && lower.rfind("www.", 0) != 0) {
    auto host = raw.substr(0, raw.find_first_of("/:?#"));
    auto dot = host.rfind('.');
    if (dot == std::string::npos) return "";
    auto suffix = u(host.substr(dot + 1));
    suffix.foldCase();
    const auto& tlds = contact_tlds();
    if (std::find(tlds.begin(), tlds.end(), utf8(suffix)) == tlds.end()) return "";
    // A title-cased suffix is commonly missing sentence spacing (noon.It).
    const auto tail = host.substr(dot + 1);
    if (tail.size() >= 2 && tail[0] >= 'A' && tail[0] <= 'Z' &&
        tail[1] >= 'a' && tail[1] <= 'z') return "";
  }
  const auto& words = contact_words(c.p.language);
  const std::string symbols = "@._-/:?=&#%";
  std::vector<std::string> pieces;
  std::string label;
  auto flush = [&] { if (!label.empty()) { pieces.push_back(label); label.clear(); } };
  for (size_t i = 0; i < raw.size(); ++i) {
    const char ch = raw[i];
    const auto symbol = symbols.find(ch);
    if (ch >= '0' && ch <= '9') {
      flush(); pieces.push_back(c.digit_words(std::string(1, ch)));
    } else if (symbol != std::string::npos || ch == '+') {
      flush(); pieces.push_back(ch == '+' ? contact_plus(c) : words[symbol]);
    } else if (scheme && i < raw.find(':')) {
      flush(); pieces.push_back(std::string(1, static_cast<char>(std::toupper(static_cast<unsigned char>(ch)))));
    } else label += ch;
  }
  flush();
  return join(pieces, " ") + ending;
}
std::string telephone(const std::string& raw, Context& c) {
  std::vector<std::string> groups;
  std::string digits;
  for (char ch : raw) {
    if (ch >= '0' && ch <= '9') digits += ch;
    else if (!digits.empty()) { groups.push_back(c.digit_words(digits)); digits.clear(); }
  }
  if (!digits.empty()) groups.push_back(c.digit_words(digits));
  return (!raw.empty() && raw[0] == '+' ? contact_plus(c) + " " : "") + join(groups, ", ");
}
int32_t russian_amount_context(const UnicodeString& text, int32_t at) {
  int32_t begin = std::max(0, at - 64);
  auto prefix = text.tempSubStringBetween(begin, at);
  int32_t leading = 0;
  while (leading < prefix.length() && u_isUWhiteSpace(prefix.charAt(leading))) ++leading;
  prefix.remove(0, leading);
  begin += leading;
  prefix.trim();
  if (prefix.endsWith(u(":"))) {
    prefix.truncate(prefix.length() - 1);
    prefix.trim();
  }
  prefix.foldCase();
  for (const auto* verb : {"составляет", "составляют", "составлял", "составляла", "составило", "составили"}) {
    const auto word = u(verb);
    const int32_t start = prefix.length() - word.length();
    if (start >= 0 && prefix.endsWith(word) &&
        (start == 0 || !u_isalpha(prefix.charAt(start - 1))))
      return begin + start;
  }
  return -1;
}
std::string run(const UnicodeString& original, Context& c) {
  UnicodeString text(original);
  for (int32_t i = 0; i < text.length(); ++i) {
    int32_t d = u_charDigitValue(text.charAt(i));
    if (d >= 0 && d <= 9) text.setCharAt(i, static_cast<char16_t>('0' + d));
    if (text.charAt(i) == 0x2212 || text.charAt(i) == 0xff0d)
      text.setCharAt(i, '-');
    if (text.charAt(i) == 0xff0e) text.setCharAt(i, '.');
    if (text.charAt(i) == 0xff0c) text.setCharAt(i, ',');
    if (text.charAt(i) == 0xff1a) text.setCharAt(i, ':');
  }
  Signals signals(text);
  std::vector<Match> selected;
  auto add = [&](int32_t a, int32_t b, std::string out) {
    for (auto& m : selected)
      if (a < m.end && b > m.begin) return;
    selected.push_back({a, b, std::move(out)});
  };
  auto scan = [&](const icu::RegexPattern& p, auto callback, bool skip_empty = false) {
    if (!may_match(&p, signals)) return;
    UErrorCode e = U_ZERO_ERROR;
    std::unique_ptr<icu::RegexMatcher> m(p.matcher(text, e));
    m->setTimeLimit(30, e);
    // Match only gaps left by higher-priority spans. A greedy number match
    // must not swallow an adjacent value merely because it overlaps a time.
    auto occupied = selected;
    std::sort(occupied.begin(), occupied.end(),
              [](const auto& a, const auto& b) { return a.begin < b.begin; });
    std::vector<std::pair<int32_t, int32_t>> gaps;
    int32_t cursor = 0;
    for (const auto& hit : occupied) {
      if (cursor < hit.begin) gaps.emplace_back(cursor, hit.begin);
      cursor = hit.end;
    }
    if (cursor < text.length()) gaps.emplace_back(cursor, text.length());
    m->useTransparentBounds(true);
    m->useAnchoringBounds(false);
    for (const auto& [begin, end] : gaps) {
      m->region(begin, end, e);
      while (m->find(e)) {
        auto group = [&](int n) { return utf8(m->group(n, e)); };
        int32_t a = m->start(e), b = m->end(e);
        std::string result = callback(group, a, b);
        if (skip_empty && result.empty()) continue;
        add(a, b,
            result.empty() ? utf8(original.tempSubStringBetween(a, b))
                           : result);
      }
    }
    if (U_FAILURE(e)) throw std::runtime_error("Voice grammar matcher limit");
  };
  auto& p = patterns();
  scan(*p.protect, [&](auto g, int32_t a, int32_t b) {
    const auto raw = g(0);
    if (electronic_shape(raw)) {
      auto spoken = electronic(raw, c);
      if (!spoken.empty()) return spoken;
    }
    // An extension such as "x123" belongs to a preceding telephone span,
    // while the same token on its own remains a protected identifier.
    int32_t before = a - 1;
    while (before >= 0 && u_isUWhiteSpace(text.charAt(before))) --before;
    if (before >= 0 && u_isdigit(text.charAt(before)) && contains_digits(raw)) {
      UErrorCode error = U_ZERO_ERROR;
      std::unique_ptr<icu::RegexMatcher> matcher(p.phone_extension->matcher(text, error));
      matcher->setTimeLimit(30, error);
      matcher->region(std::max(0, a - 64), b, error);
      matcher->useTransparentBounds(true);
      while (matcher->find(error))
        if (matcher->start(error) < a && matcher->end(error) == b)
          return std::string();
    }
    return utf8(original.tempSubStringBetween(a, b));
  }, true);
  scan(*p.date, [&](auto g, int32_t, int32_t) {
    if (!g(1).empty())
      return c.date(std::stoi(g(1)), std::stoi(g(2)), std::stoi(g(3)));
    int a = std::stoi(g(4)), b = std::stoi(g(5));
    return c.date(std::stoi(g(6)), c.p.language == "en" ? a : b,
                  c.p.language == "en" ? b : a);
  });
  scan(*p.east_date, [&](auto g, int32_t, int32_t) {
    return c.date(std::stoi(g(1)), std::stoi(g(2)), std::stoi(g(3)));
  });
  for (const auto& language : {c.p.language, std::string("en")}) {
    scan(*p.named_dates.at(language), [&](auto g, int32_t, int32_t) {
      bool day_first = !g(1).empty();
      UnicodeString name = u(g(day_first ? 2 : 4));
      name.foldCase();
      return c.date(std::stoi(g(day_first ? 3 : 6)),
                    p.month_numbers.at(utf8(name)),
                    std::stoi(g(day_first ? 1 : 5)));
    });
  }
  scan(*p.money_range_prefix, [&](auto g, int32_t start, int32_t) {
    auto a = parse_number(g(2), c.p), b = parse_number(g(4), c.p);
    if (!a || !b) return std::string();
    auto first = currency_code(g(1), c.p),
         second = g(3).empty() ? first : currency_code(g(3), c.p);
    bool accusative = c.p.language == "ru" && russian_amount_context(text, start) >= 0;
    return c.money(*a, first, accusative) + " " + range_word(c.p) + " " +
           c.money(*b, second, accusative);
  });
  scan(*p.money_range_suffix, [&](auto g, int32_t start, int32_t) {
    auto a = parse_number(g(1), c.p), b = parse_number(g(2), c.p);
    if (!a || !b) return std::string();
    auto code = currency_code(g(3), c.p);
    bool accusative = c.p.language == "ru" && russian_amount_context(text, start) >= 0;
    return c.money(*a, code, accusative) + " " + range_word(c.p) + " " + c.money(*b, code, accusative);
  });
  scan(*p.money_prefix, [&](auto g, int32_t start, int32_t) {
    auto n = parse_number(g(3), c.p);
    if (!n) return std::string();
    if (!g(1).empty()) {
      if (n->sign) return std::string();
      n->sign = g(1)[0];
    }
    return c.money(scale_money(*n, g(4)), currency_code(g(2), c.p),
                   c.p.language == "ru" && russian_amount_context(text, start) >= 0);
  });
  scan(*p.money_suffix, [&](auto g, int32_t start, int32_t) {
    auto n = parse_number(g(1), c.p);
    return n ? c.money(scale_money(*n, g(2)), currency_code(g(3), c.p),
                       c.p.language == "ru" && russian_amount_context(text, start) >= 0)
             : std::string();
  });
  scan(*p.timezone, [&](auto g, int32_t, int32_t) {
    int h = std::stoi(g(2)), m = std::stoi(g(3));
    if (h > 14 || m > 59 || (h == 14 && m != 0)) return std::string();
    return g(0).substr(0, 3) + " " + c.p.words[g(1) == "-" ? 1 : 2] + " " +
           c.count_unit(std::to_string(h), {c.p.words[5], c.p.words[6]}) + " " +
           c.count_unit(std::to_string(m), {c.p.words[7], c.p.words[8]});
  });
  scan(*p.clock_range, [&](auto g, int32_t, int32_t) {
    auto a = c.clock(std::stoi(g(1)), std::stoi(g(2)), 0, false, "");
    auto b = c.clock(std::stoi(g(3)), std::stoi(g(4)), 0, false, "");
    return a.empty() || b.empty() ? std::string()
                                  : a + " " + range_word(c.p) + " " + b;
  });
  scan(*p.hour_ampm, [&](auto g, int32_t, int32_t) {
    auto out = c.clock(std::stoi(g(1)), 0, 0, false,
                       g(2) == "A" || g(2) == "a" ? "am" : "pm");
    if (!out.empty() && g(0).back() == '.') out += '.';
    return out;
  });
  scan(*p.dotted_time, [&](auto g, int32_t, int32_t) {
    auto raw = g(0);
    auto pos = raw.find(g(1) + ".");
    auto value = c.clock(std::stoi(g(1)), std::stoi(g(2)), 0, false, "");
    return value.empty() ? std::string() : raw.substr(0, pos) + value;
  });
  scan(*p.time, [&](auto g, int32_t a, int32_t) {
    std::string ap = g(4);
    if (!ap.empty()) ap = ap == "A" || ap == "a" ? "am" : "pm";
    auto out = c.clock(
        std::stoi(g(1)), std::stoi(g(2)), g(3).empty() ? 0 : std::stoi(g(3)),
        !g(3).empty(), ap,
        (trim(utf8(text.tempSubStringBetween(std::max(0, a - 3), a))) == "o" ||
         trim(utf8(text.tempSubStringBetween(std::max(0, a - 20), a)))
                 .find("على الساعة") != std::string::npos));
    if (!out.empty() && !g(5).empty())
      out += c.p.language == "hi" ? " पर" : g(5);
    if (!out.empty() && g(0).back() == '.') out += '.';
    return out;
  });
  scan(*p.phone_extension, [&](auto g, int32_t, int32_t) {
    return telephone(g(1), c) + " " + contact_words(c.p.language)[11] + " " + c.digit_words(g(2));
  });
  scan(*p.phone, [&](auto g, int32_t, int32_t) {
    return telephone(g(0), c);
  });
  scan(*p.digit_code, [&](auto g, int32_t, int32_t) {
    auto raw = g(0);
    return raw.substr(0, raw.size() - g(1).size()) + telephone(g(1), c);
  });
  scan(*p.number_range, [&](auto g, int32_t, int32_t) {
    auto a = parse_number(g(1), c.p), b = parse_number(g(2), c.p);
    return !a || !b ? std::string()
                    : c.number(*a) + " " + range_word(c.p) + " " + c.number(*b);
  });
  scan(*p.ordinal,
       [&](auto g, int32_t, int32_t) { return c.spell(g(1), "ordinal"); });
  // A percentage between adjacent values belongs to the preceding number.
  scan(*p.percent, [&](auto g, int32_t, int32_t) {
    auto n = parse_number(g(1), c.p);
    if (!n) return std::string();
    std::string value = c.number(*n);
    return c.p.language == "zh" || c.p.language == "tr"
               ? c.p.words[3] + " " + value
               : value + " " + c.p.words[3];
  });
  scan(*p.percent_prefix, [&](auto g, int32_t, int32_t) {
    auto n = parse_number(g(1), c.p);
    if (!n) return std::string();
    auto value = c.number(*n);
    if (c.p.language == "zh") return c.p.words[3] + value;
    return c.p.language == "tr" ? c.p.words[3] + " " + value
                                : value + " " + c.p.words[3];
  });
  scan(*p.number, [&](auto g, int32_t, int32_t b) {
    auto n = parse_number(g(1), c.p);
    if (!n) {
      // A list of space-separated integers is not malformed grouping.
      auto raw = g(1);
      std::vector<std::string> words;
      for (const auto& part : split(raw, ' ')) {
        if (!digits(part)) return std::string();
        Number item{part, "", 0};
        words.push_back(c.number(item));
      }
      return join(words);
    }
    if (c.p.language == "ko" && n->fraction.empty() && n->sign == 0 &&
        n->whole.size() < 3) {
      std::string tail =
          utf8(text.tempSubStringBetween(b, std::min(b + 5, text.length())));
      tail = trim(tail);
      for (const auto& unit :
           {"개", "명", "마리", "살", "권", "잔", "대", "번"})
        if (tail.rfind(unit, 0) == 0)
          return c.spell(n->whole, "native") +
                 (b < text.length() && !u_isUWhiteSpace(text.charAt(b)) ? " "
                                                                        : "");
    }
    if (c.p.language == "zh" && n->whole == "2" && n->fraction.empty()) {
      std::string tail = trim(
          utf8(text.tempSubStringBetween(b, std::min(b + 3, text.length()))));
      for (const auto& unit : {"个", "本", "人", "只", "辆", "杯"})
        if (tail.rfind(unit, 0) == 0) return std::string("两");
    }
    return c.number(*n);
  });
  std::sort(selected.begin(), selected.end(),
            [](const auto& a, const auto& b) { return a.begin < b.begin; });
  UnicodeString out;
  int32_t pos = 0;
  for (const auto& m : selected) {
    out += original.tempSubStringBetween(pos, m.begin);
    out += u(m.output);
    pos = m.end;
  }
  out += original.tempSubString(pos);
  return utf8(out);
}
struct Cache {
  std::mutex mu;
  std::list<std::pair<std::string, std::string>> lru;
  std::unordered_map<std::string, decltype(lru)::iterator> index;
};
Cache& cache() {
  static Cache c;
  return c;
}
}  // namespace
std::string normalize(std::string_view text, std::string_view language) {
  const auto& p = profile(language);
  if (text.empty()) return std::string(text);
  UnicodeString input = u(text);
  if (utf8(input) != text) throw std::invalid_argument("Invalid UTF-8");
  bool numeric = false;
  for (int32_t i = 0; i < input.length(); ++i)
    if (u_isdigit(input.charAt(i))) {
      numeric = true;
      break;
    }
  if (!numeric && !electronic_shape(text)) return std::string(text);
  auto& ca = cache();
  std::string key = p.language + "\x1f" + std::string(text);
  bool use_cache = text.size() <= 4096;
  if (use_cache) {
    std::lock_guard<std::mutex> l(ca.mu);
    auto it = ca.index.find(key);
    if (it != ca.index.end()) {
      ca.lru.splice(ca.lru.begin(), ca.lru, it->second);
      return it->second->second;
    }
  }
  Lease lease(pool(p));
  std::string out = run(input, *lease.c);
  if (use_cache) {
    std::lock_guard<std::mutex> l(ca.mu);
    auto old = ca.index.find(key);
    if (old != ca.index.end()) {
      ca.lru.erase(old->second);
      ca.index.erase(old);
    }
    ca.lru.emplace_front(key, out);
    ca.index[key] = ca.lru.begin();
    while (ca.lru.size() > 4096) {
      ca.index.erase(ca.lru.back().first);
      ca.lru.pop_back();
    }
  }
  return out;
}
void warm() {
  patterns();
  for (const auto& p : profiles()) pool(p);
}
std::string prepare_for_segmentation(std::string_view raw,
                                     std::string_view language) {
  const auto& prof = profile(language);
  UnicodeString original = u(raw), text(original);
  if (utf8(original) != raw) throw std::invalid_argument("Invalid UTF-8");
  if (original.indexOf(static_cast<UChar>(0xffff)) >= 0)
    return std::string(raw);
  bool numeric = false;
  for (int32_t i = 0; i < text.length(); ++i) {
    int d = u_charDigitValue(text.charAt(i));
    if (d >= 0 && d <= 9) {
      text.setCharAt(i, static_cast<UChar>('0' + d));
      numeric = true;
    }
  }
  if (!numeric || original.indexOf(static_cast<UChar>(' ')) < 0)
    return std::string(raw);
  Signals signals(text);
  auto& p = patterns();
  std::vector<const icu::RegexPattern*> grammar = {
      p.date.get(),
      p.east_date.get(),
      p.money_prefix.get(),
      p.money_suffix.get(),
      p.money_range_prefix.get(),
      p.money_range_suffix.get(),
      p.clock_range.get(),
      p.number_range.get(),
      p.time.get(),
      p.dotted_time.get(),
      p.timezone.get(),
      p.phone.get(),
      p.phone_extension.get(),
      p.digit_code.get(),
      p.percent.get(),
      p.percent_prefix.get(),
      p.number.get(),
      p.hour_ampm.get(),
      p.named_dates.at(prof.language).get(),
      p.named_dates.at("en").get()};
  for (const auto* pattern : grammar) {
    if (!may_match(pattern, signals)) continue;
    UErrorCode error = U_ZERO_ERROR;
    std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(text, error));
    matcher->setTimeLimit(30, error);
    while (matcher->find(error)) {
      int32_t start = matcher->start(error), end = matcher->end(error);
      if (prof.language == "ru" &&
          (pattern == p.money_prefix.get() || pattern == p.money_suffix.get() ||
           pattern == p.money_range_prefix.get() || pattern == p.money_range_suffix.get())) {
        int32_t context = russian_amount_context(text, start);
        if (context >= 0) start = context;
      }
      if (pattern == p.number.get() &&
          !parse_number(utf8(matcher->group(1, error)), prof))
        continue;
      for (int32_t i = start; i < end; ++i)
        if (original.charAt(i) == ' ') original.setCharAt(i, 0xffff);
    }
    if (U_FAILURE(error))
      throw std::runtime_error("Voice segmentation matcher limit");
  }
  return utf8(original);
}
}  // namespace kitten_voice
extern "C" char* kitten_voice_normalize(const char* text, long long length,
                                        const char* lang,
                                        long long* out_length) {
  try {
    if (!text || length < 0 || !lang || !out_length) return nullptr;
    std::string out = kitten_voice::normalize(
        std::string_view(text, static_cast<size_t>(length)), lang);
    auto* value = static_cast<char*>(std::malloc(out.size() + 1));
    if (!value) return nullptr;
    std::memcpy(value, out.data(), out.size());
    value[out.size()] = 0;
    *out_length = static_cast<long long>(out.size());
    return value;
  } catch (...) {
    return nullptr;
  }
}
extern "C" void kitten_voice_free(void* p) { std::free(p); }
extern "C" int kitten_voice_warm() {
  try {
    kitten_voice::warm();
    return 0;
  } catch (...) {
    return 1;
  }
}
extern "C" char* kitten_voice_prepare(const char* text, long long length,
                                      const char* language,
                                      long long* out_length) {
  try {
    if (!text || length < 0 || !language || !out_length) return nullptr;
    auto out = kitten_voice::prepare_for_segmentation(
        std::string_view(text, static_cast<size_t>(length)), language);
    auto* result = static_cast<char*>(std::malloc(out.size() + 1));
    if (!result) return nullptr;
    std::memcpy(result, out.data(), out.size());
    result[out.size()] = 0;
    *out_length = static_cast<long long>(out.size());
    return result;
  } catch (...) {
    return nullptr;
  }
}
