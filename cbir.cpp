#include <algorithm>
#include <array>
#include <bits/c++config.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector2i;
using Eigen::aligned_allocator;
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

auto readData(const std::string& inputPathString) {
  fs::path inputPath(inputPathString);
  std::set<std::string> validExts{{".jpg", ".png", ".jpeg", ".tiff"}};
  std::vector<cv::Mat> X;
  std::vector<int> y;
  for (const auto &dirEntry : fs::directory_iterator(inputPath)) {
    if (!(dirEntry.is_regular_file() &&
          validExts.find(dirEntry.path().extension().string()) != validExts.end()))
      continue;
    fs::path imagePath = dirEntry.path();
    cv::Mat image = cv::imread(imagePath.string());
    if (image.channels() > 1) {
      // Convert to grayscale.
      cv::Mat imageGray;
      cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
      X.push_back(imageGray);
    }
    else
      X.push_back(image);
    int label = std::stoi(dirEntry.path().stem()) / 100;
    y.push_back(label);
  }
  return std::make_tuple(X, y);
}

template<std::size_t TABLE_SIZE = 0>
auto computeGLCMMatrix(const std::vector<cv::Mat>& Xs, const Eigen::MatrixX2i& dirs,
                       std::array<uchar, TABLE_SIZE> quantizeTable = {}) {
  for (cv::Mat X : Xs)
    if (X.depth() != CV_8U)
      throw std::invalid_argument("Image must be 8-bit grayscale");
  int nGrays;
  uchar invQuant[256];
  if constexpr (quantizeTable.empty()) {
    nGrays = 256;
  } else {
    // Get number of gray levels.
    nGrays = 1;
    for (int i = 0; i < 255; i++)
      if (quantizeTable[i] != quantizeTable[i + 1])
        nGrays++;
    // Create inverse lookup table.
    invQuant[0] = 0;
    for (int i = 1; i < 256; i++)
      if (quantizeTable[i - 1] != quantizeTable[i])
        invQuant[i] = invQuant[i - 1] + 1;
      else
        invQuant[i] = invQuant[i - 1];
  }
  std::vector<std::vector<MatrixXd, aligned_allocator<MatrixXd>>> GLCMs;
  GLCMs.reserve(Xs.size());
  // For each image in the dataset...
  for (cv::Mat X : Xs) {
    std::vector<MatrixXd, aligned_allocator<MatrixXd>> imageGLCMs;
    imageGLCMs.reserve(dirs.rows());
    // ...and for each direction...
    for (int k = 0; k < dirs.rows(); k++) {
      // ...compute the GLCM.
      MatrixXi GLCM = MatrixXi::Zero(nGrays, nGrays);
      int di = dirs(k, 0), dj = dirs(k, 1);
      int si = std::abs(std::min(di, 0)), sj = std::abs(std::min(dj, 0));
      int ei = X.rows - std::max(di, 0), ej = X.cols - std::max(dj, 0);
      for (int i = si; i < ei; i++) {
        uchar* Xi = X.ptr<uchar>(i);
        uchar* Xipd = X.ptr<uchar>(i + di);
        for (int j = sj; j < ej; j++) {
          int row = Xi[j], col = Xipd[j + dj];
          if constexpr (quantizeTable.empty())
            GLCM(row, col)++;
          else
            // Compute matrix values using indices from reverse mapping.
            GLCM(invQuant[row], invQuant[col])++;
        }
      }
      imageGLCMs.push_back(GLCM.cast<double>() / ((ei - si) * (ej - sj)));
    }
    GLCMs.push_back(imageGLCMs);
  }
  return GLCMs;
}

MatrixXd
getFeatureVectors(const std::vector<std::vector<MatrixXd,
                  aligned_allocator<MatrixXd>>>& GLCMs) {
  if (GLCMs.empty() || GLCMs[0].empty())
    throw std::invalid_argument("Empty vector of GLCMs");
  std::size_t rows = GLCMs.size(), cols = GLCMs[0].size();
  MatrixXd features(rows, 7 * cols);
  Eigen::ArrayXd arange =
      Eigen::ArrayXd::LinSpaced(GLCMs[0][0].rows(), 0, GLCMs[0][0].rows() - 1);
  Eigen::MatrixXd pairwiseDiff =
    arange.rowwise().replicate(arange.size()).rowwise() - arange.transpose();
  for (std::size_t i = 0; i < rows; i++) {
    for (std::size_t j = 0; j < cols; j++) {
      MatrixXd GLCM = GLCMs[i][j];
      // Maximum probability
      double maxProb = GLCM.maxCoeff();
      // Correlation
      double expectedRow =
          (GLCM.rowwise().sum().array() * arange).sum();
      double expectedColumn =
        (GLCM.colwise().sum().transpose().array() * arange).sum();
      Eigen::ArrayXd deviationsRow = arange - expectedRow;
      Eigen::ArrayXd deviationsColumn = arange - expectedColumn;
      double varianceRow =
        (GLCM.rowwise().sum().array() * deviationsRow.square()).sum();
      double varianceColumn =
        (GLCM.colwise().sum().transpose().array() * deviationsColumn.square()).sum();
      double correlation =
        (deviationsRow.matrix().transpose() * (GLCM * deviationsColumn.matrix()))[0] /
        std::sqrt(varianceRow * varianceColumn);
      // Contrast
      double contrast = (pairwiseDiff.array().square() * GLCM.array()).sum();
      // Uniformity
      double uniformity = GLCM.array().square().sum();
      // Homogeneity
      double homogeneity = (GLCM.array() / (1 + pairwiseDiff.array().abs())).sum();
      // Entropy
      Eigen::MatrixXd zeroCorrectedGLCM = (GLCM.array() == 0)
        .select(MatrixXd::Ones(GLCM.rows(), GLCM.cols()), GLCM);
      double entropy = -(GLCM.array() * zeroCorrectedGLCM.array().log2()).sum();
      // Dissimilarity
      double dissimilarity = (GLCM.array() * pairwiseDiff.array().abs()).sum();

      features.block<1, 7>(i, 7 * j) << maxProb, correlation, contrast,
                                        uniformity, homogeneity, entropy,
                                        dissimilarity;
    }
  }
  return features;
}

template<int QUANTIZE = 256>
MatrixXd extractFeatures(const std::vector<cv::Mat>& Xs, int dirRange) {
  // Compute directions for GLCMs.
  // int r = 0;
  // int dirRows = std::pow(2 * dirRange + 1, 2) - 1;
  // Eigen::MatrixX2i dirs(dirRows, 2);
  // for (int i = -dirRange; i <= dirRange; i++)
  //   for (int j = -dirRange; j <= dirRange; j++)
  //     dirs(r, 0) = i, dirs(r++, 1) = j;
  // Downscale X to 'QUANTIZE' levels.
  Eigen::MatrixX2i dirs(4, 2);
  dirs << 1,  1,
          1, -1,
         -1,  1,
         -1, -1;
  std::vector<cv::Mat> quantizedXs;
  quantizedXs.reserve(Xs.size());
  constexpr std::array<uchar, 256> lookupTable = [] {
    std::array<uchar, 256> table = {};
    for (int i = 0; i < 256; i++)
      table[i] = QUANTIZE * (i / QUANTIZE);
    return table;
  }();
  for (std::size_t i = 0; i < Xs.size(); i++) {
    cv::Mat quantizedX;
    cv::LUT(Xs[i], lookupTable, quantizedX);
    quantizedXs.push_back(quantizedX);
  }
  // Compute GLCM.
  std::vector<std::vector<MatrixXd, aligned_allocator<MatrixXd>>> GLCMs;
  if constexpr (QUANTIZE < 256)
    GLCMs = computeGLCMMatrix(quantizedXs, dirs, lookupTable);
  else
    GLCMs = computeGLCMMatrix(Xs, dirs);
  return getFeatureVectors(GLCMs);
}

template<bool returnPermutations = false>
auto splitTrainTest(const MatrixXd& Xs, std::vector<int> ys,
                    int nClass, int sizeClass, int sizeTrain) {
  // Wrap std::vector data in Eigen object.
  Eigen::VectorXi ysEigen = Eigen::Map<Eigen::VectorXi,
                                       Eigen::Unaligned>(ys.data(), ys.size());
  int sizeTest = sizeClass - sizeTrain;
  MatrixXd trainX(nClass * sizeTrain, Xs.cols());
  Eigen::VectorXi trainY(nClass * sizeTrain);
  MatrixXd testX(nClass * sizeTest, Xs.cols());
  Eigen::VectorXi testY(nClass * sizeTest);
  std::vector<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>> permutations;
  if constexpr (returnPermutations)
    permutations.reserve(nClass);
  std::random_device rd;
  std::mt19937 gen(rd());
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(sizeClass);
  perm.setIdentity();
  for (int i = 0; i < nClass; i++) {
    std::shuffle(perm.indices().data(),
                 perm.indices().data() + sizeClass,
                 gen);
    if constexpr (returnPermutations)
      permutations.push_back(perm);
    MatrixXd shuffledX = perm * Xs.block(i * sizeClass, 0, sizeClass, Xs.cols());
    MatrixXd trainBlock = shuffledX.block(0, 0, sizeTrain, Xs.cols());
    MatrixXd testBlock = shuffledX.block(sizeTrain, 0, sizeTest, Xs.cols());
    trainX.block(i * sizeTrain, 0,
                 sizeTrain, Xs.cols()) = trainBlock;
    testX.block(i * sizeTest, 0,
                sizeTest, Xs.cols()) = testBlock;
    // Shuffle labels along with data.
    ysEigen.segment(i * sizeClass, sizeClass) =
        perm * ysEigen.segment(i * sizeClass, sizeClass);
    trainY.segment(i * sizeTrain, sizeTrain) =
      ysEigen.segment(i * sizeClass, sizeTrain);
    testY.segment(i * sizeTest, sizeTest) =
      ysEigen.segment(sizeTrain + i * sizeClass, sizeTest);
  }
  if constexpr (returnPermutations)
    return std::make_tuple(trainX, trainY, testX, testY, permutations);
  else
    return std::make_tuple(trainX, trainY, testX, testY);
}

template<int P, bool SELECT = false>
RowMatrixXi findClosest(const MatrixXd& train, const MatrixXd& queries, int k = 5) {
  // Compute distance matrix.
  RowMatrixXi closest(queries.rows(), k);
  RowMatrixXd distances(queries.rows(), train.rows());
  for (int i = 0; i < queries.rows(); i++) {
    distances.row(i) =
      (train.rowwise() - queries.row(i)).rowwise().lpNorm<P>();
  }
  std::vector<int> idxs(distances.cols());
  int i = 0;
  for (const auto& row : distances.rowwise()) {
    auto comparator = [&row](const int &a, const int &b) {
      return row(a) < row(b);
    };
    // Generate indices.
    std::iota(idxs.begin(), idxs.end(), 0);
    if constexpr (SELECT)
      // Select first k indices using distance as comparison key.
      std::nth_element(idxs.begin(), idxs.begin() + (k - 1), idxs.end(),
                       comparator);
    else
      // Sort all indices using distance as comparison key.
      std::sort(idxs.begin(), idxs.end(), comparator);
    std::copy(idxs.begin(), idxs.begin() + k, closest.row(i++).begin());
  }
  return closest;
}

void logPerformanceMetrics(const MatrixXi& retrievedLabels,
                           const Eigen::VectorXi& testLabels,
                           int numClasses = 10) {
  Eigen::Index retrievedCols = retrievedLabels.cols();
  Eigen::Index retrievedRows = retrievedLabels.rows();
  std::initializer_list<Eigen::Index> nRetrieved =
    {10, 30, 50, 100, retrievedCols};
  for (Eigen::Index nret : nRetrieved)
    if (nret < retrievedCols) {
      MatrixXi currentRetrievedLabels =
        retrievedLabels.block(0, 0, retrievedRows, nret);
      MatrixXi currentTestLabels =
          testLabels.rowwise().replicate(nret);
      MatrixXi confusionMatrix = MatrixXi::Zero(numClasses, numClasses);
      for (int i = 0; i < currentTestLabels.rows(); i++)
        for (int j = 0; j < currentTestLabels.cols(); j++) {
          int row = currentRetrievedLabels(i, j);
          int col = currentTestLabels(i, j);
          confusionMatrix(row, col)++;
        }
      Eigen::VectorXd diag = confusionMatrix.diagonal().cast<double>();
      Eigen::VectorXd sumCWise =
        confusionMatrix.colwise().sum().cast<double>();
      Eigen::VectorXd sumRWise =
        confusionMatrix.rowwise().sum().cast<double>();
      Eigen::VectorXd recall = diag.array() / sumCWise.array();
      Eigen::VectorXd precision = diag.array() / sumRWise.array();
      std::cout << "------------------------------\n";
      std::cout << "Number retrieved per query: " << nret << " \n";
      for (int i = 0; i < numClasses; i++) {
        std::cout << "Class: " << i << '\n';
        std::cout << "\tPrecision (" << nret <<"): " << precision(i) << '\n';
        std::cout << "\tRecall (" << nret << "): " << recall(i) << '\n';
      }
      std::cout << "Number of retrieved elements: " << confusionMatrix.sum()
                << '\n';
      std::cout << "Confusion matrix:\n";
      std::cout << confusionMatrix << '\n';
      std::cout << "Average precision (" << nret << "): "
                << precision.mean() << '\n';
      std::cout << "Average recall (" << nret << "): "
                << recall.mean() << '\n';
    }
}

int main() {
  using ms = std::chrono::milliseconds;
  std::chrono::steady_clock::time_point timeStart;
  std::chrono::steady_clock::time_point timeEnd;
  int k = 100;
  std::cout << "Reading data...\n";
  timeStart = std::chrono::steady_clock::now();
  auto [X, y] = readData("data/lab4");
  timeEnd = std::chrono::steady_clock::now();
  std::cout << "Time difference = "
            << std::chrono::duration_cast<ms>(timeEnd - timeStart).count()
            << " [ms]\n";
  std::cout << "Constructing feature vectors...\n";
  timeStart = std::chrono::steady_clock::now();
  MatrixXd featureMatrix = extractFeatures<256>(X, 3);
  timeEnd = std::chrono::steady_clock::now();
  std::cout << "Time difference = "
            << std::chrono::duration_cast<ms>(timeEnd - timeStart).count()
            << " [ms]\n";
  std::cout << "Splitting into train and test sets...\n";
  timeStart = std::chrono::steady_clock::now();
  auto [trainX, trainY, testX, testY, permutations] =
    splitTrainTest<true>(featureMatrix, y, 10, 100, 70);
  timeEnd = std::chrono::steady_clock::now();
  std::cout << "Time difference = "
            << std::chrono::duration_cast<ms>(timeEnd - timeStart).count()
            << " [ms]\n";
  std::cout << "Finding closest " << k << " points...\n";
  timeStart = std::chrono::steady_clock::now();
  MatrixXi closestIndices = findClosest<1, false>(trainX, testX, k);
  timeEnd = std::chrono::steady_clock::now();
  std::cout << "Time difference = "
            << std::chrono::duration_cast<ms>(timeEnd - timeStart).count()
            << " [ms]\n";
  auto labelFetchLambda = [&trainY](int elem) { return trainY(elem); };
  MatrixXi retrievedY = closestIndices.unaryExpr(labelFetchLambda);
  // for (int i = 0; i < retrievedY.rows(); i++)
  //   std::cout << retrievedY(i, Eigen::all) << " -> " << testY[i] <<
  //   std::endl;
  logPerformanceMetrics(retrievedY, testY, 10);
  // Save some queries along with their top 5 most similar responses.
  // First, permute the images using the random permutations returned
  // by the splitTrainTest function.
  std::vector<int> idxs(X.size());
  std::iota(idxs.begin(), idxs.end(), 0);
  Eigen::VectorXi idxsEigen =
    Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(idxs.data(), idxs.size());
  for (int i = 0; i < 10; i++) {
    idxsEigen.segment<100>(i * 100) =
      permutations[i].transpose() * idxsEigen.segment<100>(i*100);
  }
  for (int i = 0; i < 3; i++) {
    int queryIdx = idxsEigen(70 + i);
    cv::Mat queryImage = X[queryIdx];
    std::ostringstream ss1;
    ss1 << "output/lab4/images/query" << i << ".jpg";
    std::string outputPathString1 = ss1.str();
    cv::imwrite(outputPathString1, queryImage);
    for (int k = 0; k < 5; k++) {
      int retrievedIdx = idxsEigen(closestIndices(i, k));
      cv::Mat retrievedImage = X[retrievedIdx];
      std::ostringstream ss2;
      ss2 << "output/lab4/images/retrieved" << i << "_" << k << ".jpg";
      std::string outputPathString2 = ss2.str();
      cv::imwrite(outputPathString2, retrievedImage);
    }
  }
  return EXIT_SUCCESS;
}
