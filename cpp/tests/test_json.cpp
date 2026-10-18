// The JSON reader: every document the Python wrote read as Python's json
// reads it, a grammar's worth of documents accepted and refused, and the
// accessors' refusals naming where they are.
#include "planetmodel/json.hpp"

#include <gtest/gtest.h>

#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include "fixtures.hpp"

namespace {

using planetmodel::Error;
using planetmodel::Json;
using planetmodel::ReadJson;

std::string Hex(std::string_view bytes) {
  auto out = std::string{};
  for (const auto c : bytes) out += std::format("{:02x}", static_cast<unsigned char>(c));
  return out;
}

/// The walk of a value, by make_fixtures.py's rule, numbers left as
/// numbers so that they compare by value.
struct Step {
  std::string path;
  std::string type;
  std::string text;
  double number = 0.0;
};

void Walk(const Json& v, std::vector<Step>& out) {
  switch (v.GetType()) {
    case Json::Type::Null: out.push_back({v.Path(), "null", ""}); break;
    case Json::Type::Bool:
      out.push_back({v.Path(), "bool", v.AsBool() ? "true" : "false"});
      break;
    case Json::Type::Number: out.push_back({v.Path(), "number", "", v.AsNumber()}); break;
    case Json::Type::String: out.push_back({v.Path(), "string", Hex(v.AsString())}); break;
    case Json::Type::Array:
      out.push_back({v.Path(), "array", std::to_string(v.Size())});
      for (const auto& e : v.AsArray()) Walk(e, out);
      break;
    case Json::Type::Object:
      out.push_back({v.Path(), "object", std::to_string(v.Size())});
      for (const auto& [key, e] : v.AsObject()) Walk(e, out);
      break;
  }
}

std::vector<std::filesystem::path> JsonFixtures() {
  auto out = std::vector<std::filesystem::path>{};
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(planetmodel::testing::FixturesDir())) {
    const auto& p = entry.path();
    if (p.extension() == ".json" && std::filesystem::exists(p.string() + ".walk")) {
      out.push_back(p);
    }
  }
  return out;
}

/// The message of the Error that reading `text` throws, or a failure.
std::string Refusal(std::string_view text) {
  try {
    static_cast<void>(ReadJson(text, "doc"));
  } catch (const Error& e) {
    return e.what();
  }
  ADD_FAILURE() << "accepted: " << text;
  return {};
}

}  // namespace

TEST(Json, ReadsWhatThePythonWroteAsThePythonReadsIt) {
  if (!std::filesystem::exists(planetmodel::testing::FixturesDir())) {
    PLANETMODEL_FIXTURE(missing, "json_corners.json");
  }
  const auto files = JsonFixtures();
  auto manifests = 0;
  for (const auto& path : files) {
    SCOPED_TRACE(path.string());
    manifests += path.string().find("manifests") != std::string::npos;
    auto steps = std::vector<Step>{};
    Walk(planetmodel::ReadJsonFile(path), steps);
    auto in = std::ifstream(path.string() + ".walk");
    auto i = std::size_t{0};
    for (auto line = std::string{}; std::getline(in, line); ++i) {
      ASSERT_LT(i, steps.size()) << "the walk is longer than the document";
      const auto tab1 = line.find('\t');
      const auto tab2 = line.find('\t', tab1 + 1);
      const auto path_ = line.substr(0, tab1);
      const auto type = line.substr(tab1 + 1, tab2 - tab1 - 1);
      const auto text = line.substr(tab2 + 1);
      EXPECT_EQ(steps[i].path, path_);
      EXPECT_EQ(steps[i].type, type) << path_;
      if (type == "number") {
        auto want = 0.0;
        std::from_chars(text.data(), text.data() + text.size(), want);
        EXPECT_EQ(steps[i].number, want) << path_;
        EXPECT_EQ(std::signbit(steps[i].number), std::signbit(want)) << path_;
      } else {
        EXPECT_EQ(steps[i].text, text) << path_;
      }
    }
    EXPECT_EQ(i, steps.size()) << "the walk is shorter than the document";
  }
  EXPECT_GE(files.size(), 3u);
  if (planetmodel::testing::FixturesRequired()) {
    EXPECT_GE(manifests, 7) << "the manifests need gmsh and PyMFEM to be written";
  }
}

TEST(Json, AcceptsTheGrammar) {
  for (const auto* text : {
           "null", "true", "false", "0", "-0", "1.5", "-1.5e-3", "1E+2", "2e10", "\"\"",
           "[]", "{}", " \t\r\n[ 1 , 2 ]\n", "{\"a\":{\"b\":[null]}}", "\"\\u0000\"",
           "\"\\ud83d\\ude00\"", "\"\xc3\xa9\"", "\"\xf0\x9f\x98\x80\"", "1e-400",
           "[1,[2,[3]]]", "{\"\":0}"}) {
    EXPECT_NO_THROW(static_cast<void>(ReadJson(text))) << text;
  }
  EXPECT_EQ(ReadJson("\"\\u00e9\"").AsString(), "\xc3\xa9");
  EXPECT_EQ(ReadJson("\"\\ud83d\\ude00\"").AsString(), "\xf0\x9f\x98\x80");
  EXPECT_TRUE(std::signbit(ReadJson("-0").AsNumber()));
  // an underflow reads as zero, keeping its sign, as Python's json reads it
  EXPECT_EQ(ReadJson("1e-400").AsNumber(), 0.0);
  EXPECT_TRUE(std::signbit(ReadJson("-1e-400").AsNumber()));
  EXPECT_EQ(ReadJson("1234.5e-330").AsNumber(), 0.0);
  EXPECT_EQ(ReadJson("0.00001e-320").AsNumber(), 0.0);
  EXPECT_EQ(ReadJson("1e-310").AsNumber(), 1e-310);
  EXPECT_EQ(ReadJson("10e307").AsNumber(), 1e308);
}

TEST(Json, RefusesWhatIsNotJsonAtItsLineAndColumn) {
  struct Case {
    const char* text;
    const char* message;
  };
  for (const auto& [text, message] : std::vector<Case>{
           // structure
           {"", "is empty"},
           {"   ", "is empty"},
           {"[1,]", "trailing comma in an array"},
           {"{\"a\":1,}", "trailing comma in an object"},
           {"[1 2]", "expected ',' or ']'"},
           {"{\"a\" 1}", "expected ':'"},
           {"{a:1}", "expected a string as an object's key"},
           {"[1] 2", "text after the end"},
           {"[", "ends where a value was expected"},
           {"{\"a\":1", "expected ',' or '}'"},
           {"\xEF\xBB\xBF{}", "byte order mark"},
           {"{\"a\":1,\"a\":2}", "\"a\" is repeated"},
           // literals
           {"nul", "expected \"null\""},
           {"True", "does not start a value"},
           {"NaN", "does not start a value"},
           {"-Infinity", "expected a digit"},
           // numbers
           {"01", "does not start with 0"},
           {"+1", "does not start a value"},
           {".5", "does not start a value"},
           {"1.", "after a decimal point"},
           {"1e", "in an exponent"},
           {"-", "expected a digit"},
           {"0x10", "text after the end"},
           {"1e400", "beyond the range"},
           {"-1e400", "beyond the range"},
           {"0.5e309", "beyond the range"},
           {"1e99999999999999999999", "beyond the range"},
           // strings
           {"\"abc", "not closed"},
           {"\"a\tb\"", "control character"},
           {"\"a\nb\"", "control character"},
           {"\"\\x\"", "is not an escape"},
           {"\"\\u12zz\"", "four hexadecimal digits"},
           {"\"\\ud800\"", "high surrogate with no low one"},
           {"\"\\ud800\\u0041\"", "high surrogate with no low one"},
           {"\"\\udc00\"", "low surrogate"},
           {"\"\xc3\"", "malformed UTF-8"},
           {"\"\xc0\xaf\"", "malformed UTF-8"},
           {"\"\xe0\x80\xaf\"", "malformed UTF-8"},
           {"\"\xed\xa0\x80\"", "malformed UTF-8"},
           {"\"\xf4\x90\x80\x80\"", "malformed UTF-8"},
           {"\"\xff\"", "malformed UTF-8"},
       }) {
    const auto got = Refusal(text);
    EXPECT_NE(got.find(message), std::string::npos) << text << " -> " << got;
    EXPECT_TRUE(std::regex_search(got, std::regex("^doc: line [0-9]+, column [0-9]+: ")))
        << got;
  }
}

TEST(Json, PlacesARefusal) {
  EXPECT_EQ(Refusal("{\n  \"a\": tru\n}"), "doc: line 2, column 8: expected \"true\"");
  EXPECT_EQ(Refusal("[1,\n 2,\n 01]"), "doc: line 3, column 3: a number does not start "
                                       "with 0 and a digit");
}

TEST(Json, NestsToItsLimitAndNoDeeper) {
  const auto depth = planetmodel::JSON_MAX_DEPTH;
  const auto within = std::string(depth, '[') + std::string(depth, ']');
  EXPECT_NO_THROW(static_cast<void>(ReadJson(within)));
  const auto beyond = std::string(depth + 1, '[') + std::string(depth + 1, ']');
  EXPECT_NE(Refusal(beyond).find("nest too deeply"), std::string::npos);
  EXPECT_NE(Refusal(std::string(100000, '[')).find("nest too deeply"), std::string::npos);
}

TEST(Json, AccessorsNameTheSourceAndThePath) {
  const auto doc = ReadJson(
      "{\"layers\": [{\"r_outer\": \"x\", \"attribute\": 1.5}], \"a b\": {\"c\": true}}",
      "earth.json");
  const auto& layer = doc.At("layers").At(0);
  EXPECT_EQ(layer.Path(), "layers[0]");
  EXPECT_EQ(doc.At("a b").At("c").Path(), "[\"a b\"].c");
  auto message = [](auto&& read) {
    try {
      read();
    } catch (const Error& e) {
      return std::string{e.what()};
    }
    return std::string{"no error"};
  };
  EXPECT_EQ(message([&] { static_cast<void>(layer.At("r_outer").AsNumber()); }),
            "earth.json: layers[0].r_outer: expected a number, got a string");
  EXPECT_EQ(message([&] { static_cast<void>(layer.At("attribute").AsInt()); }),
            "earth.json: layers[0].attribute: expected an integer, got 1.5");
  EXPECT_EQ(message([&] { static_cast<void>(layer.At("fluid")); }),
            "earth.json: layers[0]: has no member \"fluid\"");
  EXPECT_EQ(message([&] { static_cast<void>(doc.At("layers").At(3)); }),
            "earth.json: layers: has 1 elements, so no element 3");
  EXPECT_EQ(message([&] { static_cast<void>(doc.AsArray()); }),
            "earth.json: expected an array, got an object");
  EXPECT_EQ(layer.Line(), 1u);
  EXPECT_EQ(doc.Find("nothing"), nullptr);
  EXPECT_TRUE(doc.Contains("layers"));
}

TEST(Json, KnowsAnIntegerWithoutOverflowing) {
  EXPECT_TRUE(ReadJson("7").IsInteger());
  EXPECT_TRUE(ReadJson("-2147483648").IsInteger());
  EXPECT_FALSE(ReadJson("2147483648").IsInteger());
  EXPECT_FALSE(ReadJson("1e300").IsInteger());
  EXPECT_FALSE(ReadJson("2.5").IsInteger());
  EXPECT_FALSE(ReadJson("\"7\"").IsInteger());
  EXPECT_EQ(ReadJson("-3").AsInt(), -3);
}

TEST(Json, ReadsAFileAndRefusesOneThatIsNotThere) {
  EXPECT_THROW(static_cast<void>(planetmodel::ReadJsonFile("/nonexistent/x.json")), Error);
}
