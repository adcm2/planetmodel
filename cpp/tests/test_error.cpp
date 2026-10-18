#include "planetmodel/error.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(Error, NamesItsSourceAndPlace) {
  const auto e = planetmodel::Error("earth.json", "layers[2].r_outer", "is not a number");
  EXPECT_EQ(std::string{e.what()}, "earth.json: layers[2].r_outer: is not a number");
  EXPECT_EQ(e.Source(), "earth.json");
  EXPECT_EQ(e.Where(), "layers[2].r_outer");
}

TEST(Error, LeavesOutWhatItIsNotGiven) {
  EXPECT_EQ(std::string{planetmodel::Error("bare").what()}, "bare");
  EXPECT_EQ(std::string{planetmodel::Error("model.nc", "", "no part").what()},
            "model.nc: no part");
  EXPECT_EQ(std::string{planetmodel::Error("", "rho", "no layer").what()}, "rho: no layer");
}

TEST(Error, IsARuntimeError) {
  EXPECT_THROW(throw planetmodel::Error("x"), std::runtime_error);
}
