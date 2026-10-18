// Decks: the shipped mineos decks, a GIA deck and one of short layers read
// as the Python reads them, each kind of interpolant the Python's, and a
// malformed deck refused by name.
#include "planetmodel/deck.hpp"

#include <gtest/gtest.h>

#include <charconv>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures.hpp"

namespace {

using planetmodel::Deck;
using planetmodel::Error;

double ToDouble(const std::string& text) {
  auto value = 0.0;
  std::from_chars(text.data(), text.data() + text.size(), value);
  return value;
}

std::string Unhex(const std::string& hex) {
  auto out = std::string{};
  for (auto i = std::size_t{0}; i + 1 < hex.size(); i += 2) {
    out += static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16));
  }
  return out;
}

/// The deck against the Python's expectations: the layering, the header
/// and its names, and every kind's values and derivatives.
void ExpectAsThePython(const Deck& deck, const std::filesystem::path& expect, bool mineos) {
  auto in = std::ifstream(expect);
  auto values = 0;
  auto functions = std::map<std::string, planetmodel::DeckFunctions>{};
  for (const auto* kind : {"cubic", "natural", "linear"}) {
    functions.emplace(kind, planetmodel::DeckLayers(deck, kind));
  }
  const auto header = mineos ? std::optional{planetmodel::ReadMineosHeader(deck.Header(),
                                                                           deck.Source())}
                             : std::nullopt;
  const auto names = mineos ? std::optional{planetmodel::MineosNames(deck)} : std::nullopt;
  for (auto line = std::string{}; std::getline(in, line);) {
    auto words = std::istringstream{line};
    auto key = std::string{};
    words >> key;
    if (key == "knots") {
      auto n = std::size_t{0};
      words >> n;
      EXPECT_EQ(deck.NKnots(), n);
    } else if (key == "boundaries") {
      auto want = std::vector<double>{};
      for (auto w = std::string{}; words >> w;) want.push_back(ToDouble(w));
      EXPECT_EQ(deck.Boundaries(), want);
      EXPECT_EQ(functions.at("cubic").skeleton.Boundaries(), want);
    } else if (key == "header") {
      auto field = std::string{};
      auto value = std::string{};
      words >> field >> value;
      if (field == "name") {
        EXPECT_EQ(header->name, Unhex(value));
      } else if (field == "tref") {
        EXPECT_EQ(header->tref, ToDouble(value));
      } else {
        const auto counts = std::map<std::string, int>{{"ifanis", header->ifanis},
                                                       {"ifdeck", header->ifdeck},
                                                       {"nknot", header->nknot},
                                                       {"nic", header->nic},
                                                       {"noc", header->noc}};
        EXPECT_EQ(counts.at(field), std::stoi(value)) << field;
      }
    } else if (key == "extras2") {
      auto want = std::vector<std::string>{};
      for (auto w = std::string{}; words >> w;) want.push_back(w);
      EXPECT_EQ(header->extras2, want);
    } else if (key == "layer_name" || key == "interface_name") {
      auto i = std::size_t{0};
      auto name = std::string{};
      words >> i >> name;
      const auto& list = key == "layer_name" ? names->layers : names->interfaces;
      ASSERT_LT(i, list.size());
      EXPECT_EQ(list[i].value_or("-"), name) << key << " " << i;
    } else if (key == "ignored") {
      auto n = std::size_t{0};
      words >> n;
      EXPECT_EQ(names->ignored.size(), n);
    } else if (key == "absent") {
      auto i = std::size_t{0};
      auto column = std::string{};
      words >> i >> column;
      EXPECT_FALSE(functions.at("cubic").layers.at(i).contains(column)) << line;
    } else if (key == "value") {
      auto kind = std::string{};
      auto i = std::size_t{0};
      auto column = std::string{};
      auto r = std::string{};
      auto value = std::string{};
      auto derivative = std::string{};
      words >> kind >> i >> column >> r >> value >> derivative;
      const auto& f = functions.at(kind).layers.at(i).at(column);
      const auto& [lo, hi] = f.Interval();
      const auto scale = std::abs(ToDouble(value)) + 1.0;
      EXPECT_NEAR(f.Value(ToDouble(r)), ToDouble(value), 1e-12 * scale) << line;
      EXPECT_NEAR(f.Derivative(ToDouble(r)), ToDouble(derivative), 1e-10 * scale / (hi - lo))
          << line;
      ++values;
    }
  }
  EXPECT_GT(values, 20);
}

}  // namespace

TEST(Deck, ReadsTheShippedMineosDecksAsThePythonDoes) {
  for (const auto* label : {"prem200", "nocrust"}) {
    SCOPED_TRACE(label);
    PLANETMODEL_FIXTURE(path, std::string{"deck_"} + label + ".txt");
    const auto deck = planetmodel::ReadDeck(path, planetmodel::MINEOS());
    ExpectAsThePython(deck, path.parent_path() / (std::string{"deck_"} + label + ".expect"),
                      true);
  }
}

TEST(Deck, ReadsAGiaDeckWithAViscosityAbsentFromTheCore) {
  PLANETMODEL_FIXTURE(path, "deck_gia.txt");
  const auto deck = planetmodel::ReadDeck(path, planetmodel::GIA());
  ExpectAsThePython(deck, path.parent_path() / "deck_gia.expect", false);
  const auto layers = planetmodel::DeckLayers(deck);
  EXPECT_FALSE(layers.layers[0].contains("viscosity"));
  EXPECT_TRUE(layers.layers[2].contains("viscosity"));
  EXPECT_EQ(deck.Names(), planetmodel::GIA_COLUMNS);
}

TEST(Deck, JoinsShortLayersAsThePythonDoes) {
  PLANETMODEL_FIXTURE(path, "deck_knots.txt");
  const auto deck = planetmodel::ReadDeck(path, planetmodel::ColumnsFormat({"rho", "vp"}, 1));
  EXPECT_EQ(deck.NLayers(), 4u);
  ExpectAsThePython(deck, path.parent_path() / "deck_knots.expect", false);
}

TEST(Deck, ReadsNumbersAsPythonDoesAndSkipsComments) {
  const auto text = std::string{
      "# a header line\n"
      "0.0  +1.5  nan\n"
      "\n"
      "1.   2e0   NaN  # a comment\n"
      "1.0  3.0   4.0\n"
      "2.0  4.0   inf\n"};
  const auto deck =
      planetmodel::ReadDeckText(text, planetmodel::ColumnsFormat({"a", "b"}, 1), "t.txt");
  EXPECT_EQ(deck.Radius(), (std::vector<double>{0.0, 1.0, 1.0, 2.0}));
  EXPECT_EQ(deck.Column("a"), (std::vector<double>{1.5, 2.0, 3.0, 4.0}));
  EXPECT_TRUE(std::isnan(deck.Column("b")[0]));
  EXPECT_TRUE(std::isinf(deck.Column("b")[3]));
  EXPECT_EQ(deck.Boundaries(), (std::vector<double>{0.0, 1.0, 2.0}));
  const auto layers = planetmodel::DeckLayers(deck, "linear");
  EXPECT_FALSE(layers.layers[0].contains("b"));
  EXPECT_DOUBLE_EQ(layers.layers[0].at("a").Value(0.5), 1.75);
}

TEST(Deck, RefusesWhatIsNotADeckByName) {
  auto refusal = [](std::string_view text, const planetmodel::DeckFormat& format) {
    try {
      static_cast<void>(planetmodel::ReadDeckText(text, format, "d.txt"));
    } catch (const Error& e) {
      return std::string{e.what()};
    }
    return std::string{"accepted"};
  };
  const auto two = planetmodel::ColumnsFormat({"a"});
  const auto has = [](const std::string& got, std::string_view want) {
    return got.find(want) != std::string::npos;
  };
  EXPECT_PRED2(has, refusal("", two), "no knots");
  EXPECT_PRED2(has, refusal("0 1\n1 2 3\n", two), "line 2: has 3 values");
  EXPECT_PRED2(has, refusal("0 1\n1 x\n", two), "line 2: x is not a number");
  EXPECT_PRED2(has, refusal("0 1 2\n1 2 3\n", two), "1 columns after the radius, not 2");
  EXPECT_PRED2(has, refusal("1 1\n0 2\n", two), "non-decreasing");
  EXPECT_PRED2(has, refusal("0 1\n1 1\n1 1\n1 1\n2 1\n", two), "repeats more than twice");
  EXPECT_PRED2(has, refusal("0 1\n1 1\n1 1\n", two), "has one knot");
  EXPECT_PRED2(has, refusal("t\n0 1\n", planetmodel::MINEOS()), "fewer than the 3 header");
  EXPECT_PRED2(has, refusal("t\n1 1.0\n2 0 0\n0 1 2 3 4 5\n1 1 2 3 4 5\n",
                            planetmodel::MINEOS()),
               "ifanis tref ifdeck");
  EXPECT_PRED2(has, refusal("t\nr rho\n0 1 2 3 4 5 6 7\n1 1 2 3 4 5 6 7\n",
                            planetmodel::GIA()),
               "names its columns");
  const auto partial = Deck({0.0, 0.5, 1.0}, {{"a", {1.0, std::nan(""), 2.0}}});
  EXPECT_THROW(static_cast<void>(planetmodel::DeckLayers(partial)), Error);
  const auto fine = Deck({0.0, 0.5, 1.0}, {{"a", {1.0, 1.5, 2.0}}});
  EXPECT_THROW(static_cast<void>(planetmodel::DeckLayers(fine, "pchip")), Error);
  EXPECT_THROW(static_cast<void>(planetmodel::DeckLayers(fine, "quintic")), Error);
  EXPECT_THROW(static_cast<void>(planetmodel::ReadDeck("/nonexistent", two)), Error);
}

TEST(Deck, AParabolaIsAccurateBesideItsKnots) {
  // a three-knot layer under the cubic kind, read a hair from a knot
  const auto layers = planetmodel::DeckLayers(
      Deck({0.0, 0.3, 1.0}, {{"a", {1.0, 2.0, 0.5}}}), "cubic");
  const auto& f = layers.layers[0].at("a");
  const auto x = std::nextafter(0.3, 1.0);
  // p(x) = 1 + (x - 0)(d01 + (x - 0.3) d012), d01 = 10/3, d012 = (-15/7 - 10/3) / 1
  const auto d01 = 1.0 / 0.3;
  const auto d012 = (-1.5 / 0.7 - d01) / 1.0;
  EXPECT_NEAR(f.Derivative(x), d01 + (x + (x - 0.3)) * d012, 1e-12);
  EXPECT_NEAR(f.Value(0.3), 2.0, 1e-15);
}

TEST(Deck, ARadialFunctionHoldsToItsLayer) {
  const auto layers = planetmodel::DeckLayers(Deck({0.0, 0.5, 1.0}, {{"a", {1.0, 1.5, 2.0}}}));
  const auto& f = layers.layers[0].at("a");
  EXPECT_DOUBLE_EQ(f.Value(1.0 + 1e-12), 2.0);
  EXPECT_THROW(static_cast<void>(f.Value(1.0 + 1e-6)), Error);
  EXPECT_THROW(static_cast<void>(f.Value(std::nan(""))), Error);
  const auto copy = f;
  EXPECT_DOUBLE_EQ(copy.Derivative(0.25), 1.0);
}
