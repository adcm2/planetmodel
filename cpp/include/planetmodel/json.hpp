#ifndef PLANETMODEL_JSON_HPP
#define PLANETMODEL_JSON_HPP

/// @file
/// A strict reader of JSON (RFC 8259), for the manifests the Python
/// writes and any other small document a code reads.
///
/// `ReadJson(text, source)` and `ReadJsonFile(path)` parse a whole
/// document into a `Json` value or throw `planetmodel::Error` at the line
/// and column of the first fault.  Strict means: the grammar's whitespace
/// alone; numbers by the grammar (no `+1`, `.5`, `01`, hex, `NaN` or
/// `Infinity`) read by `std::from_chars`, so independent of the locale,
/// one beyond the range of a double refused; strings with every escape
/// checked, `\u` escapes and surrogate pairs encoded as UTF-8, raw
/// control characters and malformed UTF-8 refused; a key repeated in one
/// object refused; nesting deeper than `JSON_MAX_DEPTH` refused; nothing
/// after the document but whitespace.  Object members keep their order.
///
/// Every value knows where it came from: its source, its path from the
/// root (`layers[2].r_outer`, a key that is not an identifier written
/// `["a key"]`) and its line, so that an accessor asked for the wrong
/// type, a missing key or an index out of range throws an `Error` naming
/// the source and the path.

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "planetmodel/error.hpp"

namespace planetmodel {

/// How deeply arrays and objects may nest in a document.
inline constexpr auto JSON_MAX_DEPTH = 256;

/// One JSON value, with the place in its document it was read from.
class Json {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  /// An object's members, in the order the document gives them.
  using Members = std::vector<std::pair<std::string, Json>>;

  [[nodiscard]] Type GetType() const noexcept { return type_; }
  [[nodiscard]] bool IsNull() const noexcept { return type_ == Type::Null; }
  [[nodiscard]] bool IsBool() const noexcept { return type_ == Type::Bool; }
  [[nodiscard]] bool IsNumber() const noexcept { return type_ == Type::Number; }
  [[nodiscard]] bool IsString() const noexcept { return type_ == Type::String; }
  [[nodiscard]] bool IsArray() const noexcept { return type_ == Type::Array; }
  [[nodiscard]] bool IsObject() const noexcept { return type_ == Type::Object; }

  /// Whether the value is a number written as an integer, with no
  /// fraction and no exponent (as Python's json reads an `int`), that an
  /// `int` holds.
  [[nodiscard]] bool IsInteger() const noexcept {
    return type_ == Type::Number && integral_ &&
           number_ >= std::numeric_limits<int>::min() &&
           number_ <= std::numeric_limits<int>::max();
  }

  [[nodiscard]] bool AsBool() const {
    Expect(Type::Bool);
    return boolean_;
  }
  [[nodiscard]] double AsNumber() const {
    Expect(Type::Number);
    return number_;
  }
  /// The number as an `int`, refused where it was not written as an
  /// integer or lies beyond the range of one.
  [[nodiscard]] int AsInt() const {
    Expect(Type::Number);
    if (!integral_) {
      throw Error(Source(), Path(),
                  std::format("expected an integer, got {}, written with a fraction "
                              "or an exponent", number_));
    }
    if (!IsInteger()) {
      throw Error(Source(), Path(),
                  std::format("expected an integer in the range of an int, got {}",
                              number_));
    }
    return static_cast<int>(number_);
  }
  [[nodiscard]] const std::string& AsString() const {
    Expect(Type::String);
    return string_;
  }
  [[nodiscard]] const std::vector<Json>& AsArray() const {
    Expect(Type::Array);
    return array_;
  }
  [[nodiscard]] const Members& AsObject() const {
    Expect(Type::Object);
    return object_;
  }

  /// The number of elements of an array or members of an object.
  [[nodiscard]] std::size_t Size() const {
    if (type_ == Type::Array) return array_.size();
    Expect(Type::Object);
    return object_.size();
  }

  /// The member named `key` of an object, or null where it has none.
  [[nodiscard]] const Json* Find(std::string_view key) const {
    Expect(Type::Object);
    for (const auto& [name, value] : object_) {
      if (name == key) return &value;
    }
    return nullptr;
  }

  /// Whether an object has a member named `key`.
  [[nodiscard]] bool Contains(std::string_view key) const { return Find(key) != nullptr; }

  /// The member named `key` of an object, refused by name where it is
  /// missing.
  [[nodiscard]] const Json& At(std::string_view key) const {
    if (const auto* found = Find(key)) return *found;
    throw Error(Source(), Path(), std::format("has no member \"{}\"", key));
  }

  /// Element `index` of an array, refused where it is out of range.
  [[nodiscard]] const Json& At(std::size_t index) const {
    Expect(Type::Array);
    if (index >= array_.size()) {
      throw Error(Source(), Path(),
                  std::format("has {} elements, so no element {}", array_.size(), index));
    }
    return array_[index];
  }

  /// The document the value was read from, as `ReadJson` was told.
  [[nodiscard]] const std::string& Source() const noexcept {
    static const auto none = std::string{};
    return source_ ? *source_ : none;
  }

  /// The value's path from the root: empty for the root itself.
  [[nodiscard]] const std::string& Path() const noexcept { return path_; }

  /// The line of the document the value starts on, counted from one.
  [[nodiscard]] std::size_t Line() const noexcept { return line_; }

  /// The name of a type, as the errors give it.
  [[nodiscard]] static std::string_view TypeName(Type type) noexcept {
    switch (type) {
      case Type::Null: return "null";
      case Type::Bool: return "a boolean";
      case Type::Number: return "a number";
      case Type::String: return "a string";
      case Type::Array: return "an array";
      case Type::Object: return "an object";
    }
    return "a value";
  }

 private:
  friend class JsonParser;

  void Expect(Type type) const {
    if (type_ != type) {
      throw Error(Source(), Path(),
                  std::format("expected {}, got {}", TypeName(type), TypeName(type_)));
    }
  }

  Type type_ = Type::Null;
  bool boolean_ = false;
  double number_ = 0.0;
  bool integral_ = false;
  std::string string_;
  std::vector<Json> array_;
  Members object_;
  std::shared_ptr<const std::string> source_;
  std::string path_;
  std::size_t line_ = 0;
};

/// The parser behind `ReadJson`: one document, read once, the text
/// borrowed for the length of the call only.
class JsonParser {
 public:
  JsonParser(std::string_view text, std::string source)
      : text_(text), source_(std::make_shared<const std::string>(std::move(source))) {}

  Json Parse() {
    if (text_.substr(0, 3) == "\xEF\xBB\xBF") Fail("a byte order mark is not JSON");
    SkipWhitespace();
    if (AtEnd()) Fail("the document is empty");
    auto root = Value(std::string{}, 0);
    SkipWhitespace();
    if (!AtEnd()) Fail("text after the end of the document");
    return root;
  }

 private:
  [[noreturn]] void Fail(std::string_view message) const {
    throw Error(*source_, std::format("line {}, column {}", line_, pos_ - line_start_ + 1),
                message);
  }

  [[nodiscard]] bool AtEnd() const noexcept { return pos_ >= text_.size(); }
  [[nodiscard]] char Peek() const noexcept { return AtEnd() ? '\0' : text_[pos_]; }

  void SkipWhitespace() {
    while (!AtEnd()) {
      const auto c = text_[pos_];
      if (c == '\n') {
        ++pos_;
        ++line_;
        line_start_ = pos_;
      } else if (c == ' ' || c == '\t' || c == '\r') {
        ++pos_;
      } else {
        return;
      }
    }
  }

  void Literal(std::string_view word) {
    if (text_.substr(pos_, word.size()) != word) {
      Fail(std::format("expected \"{}\"", word));
    }
    pos_ += word.size();
  }

  Json Start(Json::Type type, std::string path) const {
    auto out = Json{};
    out.type_ = type;
    out.source_ = source_;
    out.path_ = std::move(path);
    out.line_ = line_;
    return out;
  }

  Json Value(std::string path, int depth) {
    switch (Peek()) {
      case '{': return Object(std::move(path), depth + 1);
      case '[': return Array(std::move(path), depth + 1);
      case '"': {
        auto out = Start(Json::Type::String, std::move(path));
        out.string_ = String();
        return out;
      }
      case 't': {
        auto out = Start(Json::Type::Bool, std::move(path));
        Literal("true");
        out.boolean_ = true;
        return out;
      }
      case 'f': {
        auto out = Start(Json::Type::Bool, std::move(path));
        Literal("false");
        return out;
      }
      case 'n': {
        auto out = Start(Json::Type::Null, std::move(path));
        Literal("null");
        return out;
      }
      default: {
        const auto c = Peek();
        if (c == '-' || (c >= '0' && c <= '9')) {
          auto out = Start(Json::Type::Number, std::move(path));
          const auto begin = pos_;
          out.number_ = Number();
          out.integral_ = text_.substr(begin, pos_ - begin).find_first_of(".eE") ==
                          std::string_view::npos;
          return out;
        }
        if (AtEnd()) Fail("the document ends where a value was expected");
        Fail(std::format("'{}' does not start a value", Printable(c)));
      }
    }
  }

  Json Array(std::string path, int depth) {
    if (depth > JSON_MAX_DEPTH) Fail("arrays and objects nest too deeply");
    auto out = Start(Json::Type::Array, path);
    ++pos_;
    SkipWhitespace();
    if (Peek() == ']') {
      ++pos_;
      return out;
    }
    while (true) {
      SkipWhitespace();
      out.array_.push_back(Value(std::format("{}[{}]", path, out.array_.size()), depth));
      SkipWhitespace();
      if (Peek() == ',') {
        ++pos_;
        SkipWhitespace();
        if (Peek() == ']') Fail("a trailing comma in an array");
        continue;
      }
      if (Peek() == ']') {
        ++pos_;
        return out;
      }
      Fail("expected ',' or ']' in an array");
    }
  }

  Json Object(std::string path, int depth) {
    if (depth > JSON_MAX_DEPTH) Fail("arrays and objects nest too deeply");
    auto out = Start(Json::Type::Object, path);
    ++pos_;
    SkipWhitespace();
    if (Peek() == '}') {
      ++pos_;
      return out;
    }
    while (true) {
      SkipWhitespace();
      if (Peek() != '"') Fail("expected a string as an object's key");
      auto key = String();
      for (const auto& member : out.object_) {
        if (member.first == key) Fail(std::format("the key \"{}\" is repeated", key));
      }
      SkipWhitespace();
      if (Peek() != ':') Fail("expected ':' after an object's key");
      ++pos_;
      SkipWhitespace();
      auto value = Value(MemberPath(path, key), depth);
      out.object_.emplace_back(std::move(key), std::move(value));
      SkipWhitespace();
      if (Peek() == ',') {
        ++pos_;
        SkipWhitespace();
        if (Peek() == '}') Fail("a trailing comma in an object");
        continue;
      }
      if (Peek() == '}') {
        ++pos_;
        return out;
      }
      Fail("expected ',' or '}' in an object");
    }
  }

  /// The path of member `key` below `path`: `.key` where the key is an
  /// identifier, `["key"]` otherwise.
  static std::string MemberPath(const std::string& path, const std::string& key) {
    auto identifier = !key.empty() && !(key[0] >= '0' && key[0] <= '9');
    for (const auto c : key) {
      identifier = identifier && ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                  (c >= '0' && c <= '9') || c == '_');
    }
    if (identifier) return path.empty() ? key : std::format("{}.{}", path, key);
    auto quoted = std::string{};
    for (const auto c : key) {
      if (c == '"' || c == '\\') quoted += '\\';
      quoted += c;
    }
    return std::format("{}[\"{}\"]", path, quoted);
  }

  double Number() {
    const auto begin = pos_;
    if (Peek() == '-') ++pos_;
    if (Peek() == '0') {
      ++pos_;
      if (Peek() >= '0' && Peek() <= '9') Fail("a number does not start with 0 and a digit");
    } else if (Peek() >= '1' && Peek() <= '9') {
      while (Peek() >= '0' && Peek() <= '9') ++pos_;
    } else {
      Fail("expected a digit in a number");
    }
    if (Peek() == '.') {
      ++pos_;
      if (!(Peek() >= '0' && Peek() <= '9')) Fail("expected a digit after a decimal point");
      while (Peek() >= '0' && Peek() <= '9') ++pos_;
    }
    if (Peek() == 'e' || Peek() == 'E') {
      ++pos_;
      if (Peek() == '+' || Peek() == '-') ++pos_;
      if (!(Peek() >= '0' && Peek() <= '9')) Fail("expected a digit in an exponent");
      while (Peek() >= '0' && Peek() <= '9') ++pos_;
    }
    const auto digits = text_.substr(begin, pos_ - begin);
    auto value = 0.0;
    const auto [end, error] =
        std::from_chars(digits.data(), digits.data() + digits.size(), value);
    if (error == std::errc::result_out_of_range) {
      if (DecimalOrder(digits) < 0) return digits[0] == '-' ? -0.0 : 0.0;
      pos_ = begin;
      Fail(std::format("{} is beyond the range of a double", digits));
    }
    if (error != std::errc{} || end != digits.data() + digits.size()) {
      pos_ = begin;
      Fail(std::format("{} is not a number", digits));
    }
    return value;
  }

  /// The power of ten of a number's leading significant digit, from its
  /// text by the grammar: negative for a number below one in magnitude,
  /// so that an underflow (read as zero, as Python's json reads it) is
  /// told from an overflow (refused).
  static long long DecimalOrder(std::string_view digits) {
    auto mantissa = digits.substr(digits[0] == '-' ? 1 : 0);
    auto exponent = 0LL;
    if (const auto e = mantissa.find_first_of("eE"); e != std::string_view::npos) {
      auto text = mantissa.substr(e + 1);
      if (!text.empty() && text[0] == '+') text.remove_prefix(1);
      if (std::from_chars(text.data(), text.data() + text.size(), exponent).ec !=
          std::errc{}) {
        exponent = text[0] == '-' ? std::numeric_limits<long long>::min() / 2
                                  : std::numeric_limits<long long>::max() / 2;
      }
      mantissa = mantissa.substr(0, e);
    }
    const auto point = mantissa.find('.');
    const auto whole = mantissa.substr(0, point);
    if (const auto lead = whole.find_first_not_of('0'); lead != std::string_view::npos) {
      return exponent + static_cast<long long>(whole.size() - lead) - 1;
    }
    const auto fraction = point == std::string_view::npos ? std::string_view{}
                                                          : mantissa.substr(point + 1);
    const auto lead = fraction.find_first_not_of('0');
    return exponent - static_cast<long long>(lead) - 1;
  }

  std::uint32_t Hex4() {
    auto code = std::uint32_t{0};
    for (auto i = 0; i < 4; ++i) {
      const auto c = Peek();
      auto digit = 0;
      if (c >= '0' && c <= '9') {
        digit = c - '0';
      } else if (c >= 'a' && c <= 'f') {
        digit = c - 'a' + 10;
      } else if (c >= 'A' && c <= 'F') {
        digit = c - 'A' + 10;
      } else {
        Fail("expected four hexadecimal digits after \\u");
      }
      code = code * 16 + static_cast<std::uint32_t>(digit);
      ++pos_;
    }
    return code;
  }

  static void AppendUtf8(std::string& out, std::uint32_t code) {
    if (code < 0x80) {
      out += static_cast<char>(code);
    } else if (code < 0x800) {
      out += static_cast<char>(0xC0 | (code >> 6));
      out += static_cast<char>(0x80 | (code & 0x3F));
    } else if (code < 0x10000) {
      out += static_cast<char>(0xE0 | (code >> 12));
      out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (code & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (code >> 18));
      out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (code & 0x3F));
    }
  }

  /// One raw UTF-8 sequence copied through, checked as Unicode allows
  /// it: no overlong form, no surrogate, nothing beyond U+10FFFF.
  void Utf8Sequence(std::string& out) {
    const auto lead = static_cast<unsigned char>(text_[pos_]);
    auto length = 0;
    auto code = std::uint32_t{0};
    auto least = std::uint32_t{0};
    if (lead >= 0xC2 && lead <= 0xDF) {
      length = 2, code = lead & 0x1F, least = 0x80;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      length = 3, code = lead & 0x0F, least = 0x800;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
      length = 4, code = lead & 0x07, least = 0x10000;
    } else {
      Fail("malformed UTF-8 in a string");
    }
    for (auto i = 1; i < length; ++i) {
      const auto at = pos_ + static_cast<std::size_t>(i);
      const auto next = at < text_.size() ? static_cast<unsigned char>(text_[at]) : 0u;
      if ((next & 0xC0) != 0x80) Fail("malformed UTF-8 in a string");
      code = (code << 6) | (next & 0x3F);
    }
    if (code < least || (code >= 0xD800 && code <= 0xDFFF) || code > 0x10FFFF) {
      Fail("malformed UTF-8 in a string");
    }
    out.append(text_.substr(pos_, static_cast<std::size_t>(length)));
    pos_ += static_cast<std::size_t>(length);
  }

  std::string String() {
    ++pos_;  // the opening quote
    auto out = std::string{};
    while (true) {
      if (AtEnd()) Fail("a string is not closed");
      const auto c = text_[pos_];
      if (c == '"') {
        ++pos_;
        return out;
      }
      if (static_cast<unsigned char>(c) < 0x20) {
        Fail("a control character in a string must be escaped");
      }
      if (static_cast<unsigned char>(c) >= 0x80) {
        Utf8Sequence(out);
        continue;
      }
      if (c != '\\') {
        out += c;
        ++pos_;
        continue;
      }
      ++pos_;
      const auto e = Peek();
      ++pos_;
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          auto code = Hex4();
          if (code >= 0xDC00 && code <= 0xDFFF) Fail("a low surrogate with no high one");
          if (code >= 0xD800 && code <= 0xDBFF) {
            if (text_.substr(pos_, 2) != "\\u") Fail("a high surrogate with no low one");
            pos_ += 2;
            const auto low = Hex4();
            if (!(low >= 0xDC00 && low <= 0xDFFF)) Fail("a high surrogate with no low one");
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
          }
          AppendUtf8(out, code);
          break;
        }
        default:
          --pos_;
          Fail(std::format("\\{} is not an escape", Printable(e)));
      }
    }
  }

  /// A character as an error shows it: itself where printable, else its
  /// code.
  static std::string Printable(char c) {
    const auto u = static_cast<unsigned char>(c);
    return (u >= 0x20 && u < 0x7F) ? std::string(1, c) : std::format("\\x{:02x}", u);
  }

  std::string_view text_;
  std::shared_ptr<const std::string> source_;
  std::size_t pos_ = 0;
  std::size_t line_ = 1;
  std::size_t line_start_ = 0;
};

/// The document in `text`, read whole; `source` names it in errors (a
/// path, say).
[[nodiscard]] inline Json ReadJson(std::string_view text, std::string source = "<text>") {
  return JsonParser(text, std::move(source)).Parse();
}

/// The document in the file at `path`.
[[nodiscard]] inline Json ReadJsonFile(const std::filesystem::path& path) {
  auto in = std::ifstream(path, std::ios::binary);
  if (!in) throw Error(path.string(), "", "cannot be opened");
  const auto text = std::string(std::istreambuf_iterator<char>(in), {});
  return ReadJson(text, path.string());
}

}  // namespace planetmodel

#endif  // PLANETMODEL_JSON_HPP
