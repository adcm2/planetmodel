#ifndef PLANETMODEL_DECK_HPP
#define PLANETMODEL_DECK_HPP

/// @file
/// Decks, as `planetmodel.deck` reads them: tables of knots, radius first
/// and one column per named field, in which a repeated radius marks a
/// boundary.
///
/// A `DeckFormat` names the columns (by row width, where a format admits
/// more than one) and says how many header lines precede the table and
/// what they must say.  `MINEOS()` is the format of the mineos and PREM
/// decks, three header lines (a title; `ifanis tref ifdeck`; `nknot nic
/// noc`) and nine or six columns; `GIA()` that of a viscoelastic model for
/// glacial isostatic adjustment, a title and the line naming the columns,
/// `r rho vpv vsv vph vsh eta viscosity` in SI, NaN where a layer lacks a
/// field.  `ReadDeck` gives a `Deck`, the numbers and the header lines,
/// nothing else; the table's blank lines and `#` comments are skipped and
/// its numbers read as Python reads them, `nan` and a leading `+`
/// included.  `MineosNames` gives the layer and interface names a mineos
/// header implies, and `DeckLayers` each layer's columns as functions of
/// the radius, joined by the kind asked for, on the deck's skeleton.

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "planetmodel/error.hpp"
#include "planetmodel/radial.hpp"
#include "planetmodel/skeleton.hpp"

namespace planetmodel {

/// The columns of a transversely isotropic mineos deck, after the radius.
inline const auto MINEOS_COLUMNS = std::vector<std::string>{
    "rho", "vpv", "vsv", "qkappa", "qmu", "vph", "vsh", "eta"};

/// The columns of an isotropic mineos deck, after the radius.
inline const auto MINEOS_ISOTROPIC_COLUMNS =
    std::vector<std::string>{"rho", "vp", "vs", "qkappa", "qmu"};

/// The columns of a GIA deck, after the radius.
inline const auto GIA_COLUMNS = std::vector<std::string>{
    "rho", "vpv", "vsv", "vph", "vsh", "eta", "viscosity"};

/// What a family of deck files looks like.
struct DeckFormat {
  /// The format's name, as errors give it.
  std::string name = "custom";
  /// The column names after the radius, by the number of them a row has.
  std::map<std::size_t, std::vector<std::string>> columns{};
  /// How many lines precede the table.
  std::size_t header_lines = 0;
  /// What the header lines must say: throws `Error` naming `source`
  /// where they do not; none for a header kept verbatim.
  std::function<void(const std::vector<std::string>& lines, const std::string& source)>
      check_header{};

  /// The column names of a row with `width` values after the radius.
  [[nodiscard]] const std::vector<std::string>& Names(std::size_t width,
                                                      const std::string& source) const {
    if (const auto it = columns.find(width); it != columns.end()) return it->second;
    auto widths = std::string{};
    for (const auto& [w, names] : columns) widths += (widths.empty() ? "" : " or ") +
                                                     std::to_string(w);
    throw Error(source, "",
                std::format("a {} deck has {} columns after the radius, not {}", name,
                            widths, width));
  }
};

/// The three header lines of a mineos deck, read.
struct MineosHeader {
  std::string name;
  int ifanis = 0;
  double tref = 0.0;
  int ifdeck = 0;
  int nknot = 0;
  int nic = 0;
  int noc = 0;
  /// The tokens after the three each of the second and third lines carry.
  std::vector<std::string> extras2{};
  std::vector<std::string> extras3{};
};

namespace detail {

inline std::vector<std::string> Tokens(std::string_view line) {
  auto in = std::istringstream{std::string{line}};
  auto out = std::vector<std::string>{};
  for (auto token = std::string{}; in >> token;) out.push_back(token);
  return out;
}

inline std::string Trimmed(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) return {};
  const auto last = text.find_last_not_of(" \t\r\n");
  return std::string{text.substr(first, last - first + 1)};
}

/// A number as Python's float reads a deck's: a leading sign, `nan` and
/// `inf` included; empty where the token is not one.
inline std::optional<double> Number(std::string_view token) {
  if (!token.empty() && token.front() == '+') token.remove_prefix(1);
  auto value = 0.0;
  const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
  if (error != std::errc{} || end != token.data() + token.size()) return std::nullopt;
  return value;
}

inline std::optional<int> Integer(std::string_view token) {
  if (!token.empty() && token.front() == '+') token.remove_prefix(1);
  auto value = 0;
  const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
  if (error != std::errc{} || end != token.data() + token.size()) return std::nullopt;
  return value;
}

}  // namespace detail

/// The mineos header of lines `lines` read from `source`, or an `Error`
/// saying what the second and third lines must carry.
[[nodiscard]] inline MineosHeader ReadMineosHeader(const std::vector<std::string>& lines,
                                                   const std::string& source) {
  if (lines.size() < 3) {
    throw Error(source, "header", "a mineos deck has three header lines");
  }
  const auto h2 = detail::Tokens(lines.at(1));
  const auto h3 = detail::Tokens(lines.at(2));
  auto fail = [&] {
    throw Error(source, "header",
                "a mineos deck's second and third header lines carry `ifanis tref "
                "ifdeck` and `nknot nic noc`");
  };
  if (h2.size() < 3 || h3.size() < 3) fail();
  const auto ifanis = detail::Integer(h2[0]);
  const auto tref = detail::Number(h2[1]);
  const auto ifdeck = detail::Integer(h2[2]);
  const auto nknot = detail::Integer(h3[0]);
  const auto nic = detail::Integer(h3[1]);
  const auto noc = detail::Integer(h3[2]);
  if (!ifanis || !tref || !ifdeck || !nknot || !nic || !noc) fail();
  return {.name = detail::Trimmed(lines[0]),
          .ifanis = *ifanis,
          .tref = *tref,
          .ifdeck = *ifdeck,
          .nknot = *nknot,
          .nic = *nic,
          .noc = *noc,
          .extras2 = std::vector<std::string>(h2.begin() + 3, h2.end()),
          .extras3 = std::vector<std::string>(h3.begin() + 3, h3.end())};
}

/// The mineos format: three header lines, then `r` and the eight columns
/// of `MINEOS_COLUMNS` or the five of `MINEOS_ISOTROPIC_COLUMNS`.
[[nodiscard]] inline const DeckFormat& MINEOS() {
  static const auto format = DeckFormat{
      .name = "mineos",
      .columns = {{8, MINEOS_COLUMNS}, {5, MINEOS_ISOTROPIC_COLUMNS}},
      .header_lines = 3,
      .check_header = [](const std::vector<std::string>& lines, const std::string& source) {
        static_cast<void>(ReadMineosHeader(lines, source));
      }};
  return format;
}

/// The GIA format: a title line, the line naming the columns, then `r`
/// and the columns of `GIA_COLUMNS`.
[[nodiscard]] inline const DeckFormat& GIA() {
  static const auto format = DeckFormat{
      .name = "gia",
      .columns = {{7, GIA_COLUMNS}},
      .header_lines = 2,
      .check_header = [](const std::vector<std::string>& lines, const std::string& source) {
        auto want = std::vector<std::string>{"r"};
        want.insert(want.end(), GIA_COLUMNS.begin(), GIA_COLUMNS.end());
        if (detail::Tokens(lines.at(1)) != want) {
          throw Error(source, "header",
                      std::format("a GIA deck's second line names its columns, r rho vpv "
                                  "vsv vph vsh eta viscosity; got {}",
                                  detail::Trimmed(lines.at(1))));
        }
      }};
  return format;
}

/// A table of knots: radii, one column per name, and the header lines.
class Deck {
 public:
  /// A deck of `radius`, non-decreasing with every repeat a boundary, and
  /// `columns` each of one value per knot (NaN where absent); its layers
  /// are checked as it is built.
  Deck(std::vector<double> radius,
       std::vector<std::pair<std::string, std::vector<double>>> columns,
       std::vector<std::string> header = {}, std::string source = "the deck")
      : radius_(std::move(radius)),
        columns_(std::move(columns)),
        header_(std::move(header)),
        source_(std::move(source)) {
    if (radius_.size() < 2) Fail("needs at least two knot radii");
    for (auto i = std::size_t{1}; i < radius_.size(); ++i) {
      if (!(radius_[i] >= radius_[i - 1])) Fail("knot radii must be non-decreasing");
    }
    for (const auto& [name, values] : columns_) {
      if (values.size() != radius_.size()) {
        Fail(std::format("column {} has {} values for {} radii", name, values.size(),
                         radius_.size()));
      }
    }
    static_cast<void>(Layers());
  }

  [[nodiscard]] const std::vector<double>& Radius() const noexcept { return radius_; }
  [[nodiscard]] std::size_t NKnots() const noexcept { return radius_.size(); }
  /// The header lines, as read.
  [[nodiscard]] const std::vector<std::string>& Header() const noexcept { return header_; }
  /// Where the deck was read from.
  [[nodiscard]] const std::string& Source() const noexcept { return source_; }

  /// The column names, in the deck's order.
  [[nodiscard]] std::vector<std::string> Names() const {
    auto out = std::vector<std::string>{};
    for (const auto& [name, values] : columns_) out.push_back(name);
    return out;
  }
  [[nodiscard]] bool HasColumn(std::string_view name) const {
    for (const auto& [n, values] : columns_) {
      if (n == name) return true;
    }
    return false;
  }
  /// The values of column `name` at every knot.
  [[nodiscard]] const std::vector<double>& Column(std::string_view name) const {
    for (const auto& [n, values] : columns_) {
      if (n == name) return values;
    }
    throw Error(source_, "", std::format("has no column {}", name));
  }

  /// The rows of every layer, centre outward, as [begin, end): one span
  /// between repeated radii, each of at least two strictly increasing
  /// knots.
  [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> Layers() const {
    auto out = std::vector<std::pair<std::size_t, std::size_t>>{};
    auto start = std::size_t{0};
    for (auto i = std::size_t{1}; i <= radius_.size(); ++i) {
      if (i < radius_.size() && radius_[i] != radius_[i - 1]) continue;
      if (i - start < 2) {
        if (i < radius_.size() && i >= 2 && radius_[i - 1] == radius_[i - 2]) {
          Fail("a radius repeats more than twice: a layer of no thickness");
        }
        Fail(std::format("the layer starting at r = {} has one knot; every layer needs "
                         "at least two",
                         radius_[start]));
      }
      out.emplace_back(start, i);
      start = i;
    }
    return out;
  }
  [[nodiscard]] std::size_t NLayers() const { return Layers().size(); }

  /// The skeleton's boundary radii: the ends and every repeated radius.
  [[nodiscard]] std::vector<double> Boundaries() const {
    auto out = std::vector<double>{radius_.front()};
    for (auto i = std::size_t{1}; i < radius_.size(); ++i) {
      if (radius_[i] == radius_[i - 1]) out.push_back(radius_[i]);
    }
    out.push_back(radius_.back());
    return out;
  }

 private:
  [[noreturn]] void Fail(std::string_view message) const { throw Error(source_, "", message); }

  std::vector<double> radius_;
  std::vector<std::pair<std::string, std::vector<double>>> columns_;
  std::vector<std::string> header_;
  std::string source_;
};

/// The deck in `text`, by `format`, `source` naming it in errors.
[[nodiscard]] inline Deck ReadDeckText(std::string_view text, const DeckFormat& format,
                                       std::string source = "the deck") {
  auto lines = std::vector<std::string>{};
  {
    auto in = std::istringstream{std::string{text}};
    for (auto line = std::string{}; std::getline(in, line);) lines.push_back(line);
  }
  if (lines.size() < format.header_lines) {
    throw Error(source, "",
                std::format("{} lines, fewer than the {} header lines of a {} deck",
                            lines.size(), format.header_lines, format.name));
  }
  const auto header = std::vector<std::string>(
      lines.begin(), lines.begin() + static_cast<std::ptrdiff_t>(format.header_lines));
  if (format.check_header) format.check_header(header, source);

  auto rows = std::vector<std::vector<double>>{};
  for (auto i = format.header_lines; i < lines.size(); ++i) {
    auto line = std::string_view{lines[i]};
    line = line.substr(0, line.find('#'));
    const auto tokens = detail::Tokens(line);
    if (tokens.empty()) continue;
    auto row = std::vector<double>{};
    for (const auto& token : tokens) {
      const auto value = detail::Number(token);
      if (!value) {
        throw Error(source, std::format("line {}", i + 1),
                    std::format("{} is not a number", token));
      }
      row.push_back(*value);
    }
    if (!rows.empty() && row.size() != rows.front().size()) {
      throw Error(source, std::format("line {}", i + 1),
                  std::format("has {} values, the rows before it {}", row.size(),
                              rows.front().size()));
    }
    rows.push_back(std::move(row));
  }
  if (rows.empty()) throw Error(source, "", "no knots after the header");
  const auto& names = format.Names(rows.front().size() - 1, source);
  auto radius = std::vector<double>{};
  auto columns = std::vector<std::pair<std::string, std::vector<double>>>{};
  for (const auto& name : names) columns.emplace_back(name, std::vector<double>{});
  for (const auto& row : rows) {
    radius.push_back(row[0]);
    for (auto j = std::size_t{0}; j < names.size(); ++j) columns[j].second.push_back(row[j + 1]);
  }
  return Deck(std::move(radius), std::move(columns), header, std::move(source));
}

/// The deck in the file at `path`, by `format`.
[[nodiscard]] inline Deck ReadDeck(const std::filesystem::path& path, const DeckFormat& format) {
  auto in = std::ifstream(path, std::ios::binary);
  if (!in) throw Error(path.string(), "", "cannot be opened");
  const auto text = std::string(std::istreambuf_iterator<char>(in), {});
  return ReadDeckText(text, format, path.string());
}

/// A format of the given column names after the radius, its header
/// `header_lines` lines kept as they are.
[[nodiscard]] inline DeckFormat ColumnsFormat(std::vector<std::string> columns,
                                              std::size_t header_lines = 0) {
  const auto width = columns.size();
  return DeckFormat{.columns = {{width, std::move(columns)}}, .header_lines = header_lines};
}

/// The names a mineos header implies, and what in it did not hold.
struct DeckNames {
  /// One per layer, empty where the header names none.
  std::vector<std::optional<std::string>> layers{};
  /// One per interface, centre outward.
  std::vector<std::optional<std::string>> interfaces{};
  /// What of the header was ignored, and why, as the Python warns it.
  std::vector<std::string> ignored{};
};

/// The layer and interface names a mineos header implies: `nic` and
/// `noc` count the knots of the inner core and to the top of the outer
/// core, so where each sits on a repeated radius the first two layers are
/// `inner_core` and `outer_core` under `icb` and `cmb`; the outermost
/// interface is `surface`.  A count of zero says there is no such core; a
/// count that does not sit on a repeated radius, or a knot count other
/// than the table's, is ignored and said so in `ignored`.
[[nodiscard]] inline DeckNames MineosNames(const Deck& deck) {
  const auto h = ReadMineosHeader(deck.Header(), deck.Source());
  const auto& r = deck.Radius();
  const auto n = deck.NLayers();
  const auto first = r.front() > 0.0 ? std::size_t{1} : std::size_t{0};
  auto out = DeckNames{.layers = std::vector<std::optional<std::string>>(n),
                       .interfaces = std::vector<std::optional<std::string>>(n + first)};
  out.interfaces.back() = "surface";
  const auto boundaries = deck.Boundaries();
  const auto cores = std::array<std::tuple<int, std::string_view, std::string_view,
                                           std::string_view>, 2>{
      {{h.nic, "nic", "inner_core", "icb"}, {h.noc, "noc", "outer_core", "cmb"}}};
  for (const auto& [count, key, layer, face] : cores) {
    const auto i = static_cast<std::size_t>(count < 0 ? 0 : count);
    if (!(count >= 1 && i < r.size() && r[i - 1] == r[i])) {
      if (count != 0) {
        out.ignored.push_back(std::format("the header's {} = {} does not sit on a repeated "
                                          "radius; the layer is left unnamed",
                                          key, count));
      }
      continue;
    }
    const auto k = static_cast<std::size_t>(
        std::lower_bound(boundaries.begin(), boundaries.end(), r[i]) - boundaries.begin());
    if (k > 0 && k < n) {
      out.layers[k - 1] = std::string{layer};
      out.interfaces[k - 1 + first] = std::string{face};
    }
  }
  if (h.nknot != static_cast<int>(deck.NKnots())) {
    out.ignored.push_back(std::format("the header says {} knots, the table has {}; the table "
                                      "is taken as the truth",
                                      h.nknot, deck.NKnots()));
  }
  return out;
}

/// A deck's skeleton and each layer's columns as functions of the radius.
struct DeckFunctions {
  Skeleton skeleton;
  /// One map of name to function per layer; a column NaN throughout a
  /// layer is absent from it.
  std::vector<std::map<std::string, RadialFunction>> layers{};
};

/// The kinds a deck's knots may be joined by in C++, by name.
[[nodiscard]] inline KnotKind KnotKindNamed(std::string_view kind) {
  if (kind == "cubic") return KnotKind::Cubic;
  if (kind == "natural") return KnotKind::Natural;
  if (kind == "linear") return KnotKind::Linear;
  if (kind == "pchip" || kind == "akima") {
    throw Error("DeckLayers", "kind",
                std::format("{} is the Python's alone; in C++ the kinds are cubic, natural "
                            "and linear", kind));
  }
  throw Error("DeckLayers", "kind",
              std::format("{} is not a kind; they are cubic, natural and linear", kind));
}

/// The skeleton of a deck and every layer's columns, each the piecewise
/// polynomial of `kind` through the layer's knots, on its layer with the
/// skeleton's default tolerance.  A column NaN on part of a layer is
/// refused.
[[nodiscard]] inline DeckFunctions DeckLayers(const Deck& deck,
                                              std::string_view kind = "cubic") {
  const auto knot_kind = KnotKindNamed(kind);
  auto out = DeckFunctions{.skeleton = Skeleton(deck.Boundaries()), .layers = {}};
  const auto tolerance = DEFAULT_RTOL * out.skeleton.Span();
  const auto& r = deck.Radius();
  const auto slices = deck.Layers();
  for (auto i = std::size_t{0}; i < slices.size(); ++i) {
    const auto [begin, end] = slices[i];
    auto fields = std::map<std::string, RadialFunction>{};
    for (const auto& name : deck.Names()) {
      const auto& values = deck.Column(name);
      auto knots = std::vector<double>(r.begin() + static_cast<std::ptrdiff_t>(begin),
                                       r.begin() + static_cast<std::ptrdiff_t>(end));
      auto y = std::vector<double>(values.begin() + static_cast<std::ptrdiff_t>(begin),
                                   values.begin() + static_cast<std::ptrdiff_t>(end));
      const auto missing = std::ranges::count_if(y, [](double v) { return std::isnan(v); });
      if (missing == static_cast<std::ptrdiff_t>(y.size())) continue;
      if (missing > 0) {
        throw Error(deck.Source(), name,
                    std::format("is NaN on part of layer {}; a column is absent from a "
                                "layer only when NaN throughout it",
                                i));
      }
      fields.emplace(name, ThroughKnots(std::move(knots), std::move(y), knot_kind, tolerance));
    }
    out.layers.push_back(std::move(fields));
  }
  return out;
}

}  // namespace planetmodel

#endif  // PLANETMODEL_DECK_HPP
