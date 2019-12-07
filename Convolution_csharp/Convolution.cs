using System;
using System.Collections.Generic;
using System.Text;

namespace Convolution_csharp
{
  class Convolution
  {
    //only accepting square input and square filter
    int[,] input = new int[5, 5] { { 0, 1, 0, 2, 1 }, { 2, 2, 2, 0, 1 }, { 0, 1, 2, 1, 2 }, { 2, 1, 1, 1, 0 }, { 2, 1, 2, 1, 0 } };
    int[,] filterOne = new int[3, 3] { { -1, 0, -1 }, { -1, -1, -1 }, { 1, 0, -1 } };
    int[,] filterTwo = new int[3, 3] { { 0, -1, 0 }, { -1, 0, 0 }, { 1, 0, 1 } };
    int[,] filterThree = new int[3, 3] { { -1, 0, 1 }, { -1, 1, -1 }, { 1, 0, 1 } };
    List<int[,]> filters = new List<int[,]>();
    int inputSize = 0;
    int filterSize = 0;
    int outputSize = 0;
    int stride = 2;
    int zeroPadding = 0;
    int[,] paddedInput;
    int[,] bias = new int[3, 3] { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };
    //int[,] bias = new int[3, 3] { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0, } };
    List<int[,]> convolvedMatrices = new List<int[,]>();
    int[,] withBias;

    public Convolution()
    {
      filters.Add(filterOne);
      filters.Add(filterTwo);
      filters.Add(filterThree);
      inputSize = input.GetLength(0);
      paddedInput = input;
      filterSize = filterOne.GetLength(0);
      //we are requiring output size to equal filter size
      outputSize = filterSize;
      withBias = new int[outputSize, outputSize];
    }

    public Convolution convolve()
    {
      setStrideAndPadding();
      padInput();
      createConvolvedMatrices();
      //Add all convolved matrices
      var convolvedSum = addMatrices(convolvedMatrices);
      withBias = addMatrix(convolvedSum, bias);
      return this;
    }

    void setStrideAndPadding()
    {
      //find stride and zeroPadding where (inputSize - filterSize + 2 * zeroPadding) / stride + 1 is an integer
      //iterate incrementally over strides while iterating incrementally over zeroPadding during each stride iteration
      bool leave = false;
      for (int i = stride; i >= 0; i--)
      {
        int t;
        for (t = zeroPadding; t <= zeroPadding + 3; t++)
        {
          int strideSpots = inputSize - filterSize + 2 * t;
          if (strideSpots >= 0 && strideSpots % i == 0 && strideSpots / i + 1 == outputSize)
          {
            stride = i;
            zeroPadding = t;
            leave = true;
            break;
          }
        }
        if (leave)
          break;
      }
      Console.WriteLine("stride: " + stride + "\nzeroPadding: " + zeroPadding);
    }

    void padInput()
    {
      if(zeroPadding > 0)
      {
        var paddedDim = inputSize + 2 * zeroPadding;
        paddedInput = new int[paddedDim, paddedDim];
        //copy in original input
        for (int i = 0; i < inputSize * inputSize; i++)
        {
          var row = zeroPadding + i / inputSize;
          var column = zeroPadding + i % inputSize;
          paddedInput[row, column] = input[i / inputSize, i % inputSize];
        }
      }
    }

    void createConvolvedMatrices()
    {
      //Perform the convolutions
      //loop over filters
      convolvedMatrices = new List<int[,]>();
      foreach (var filter in filters)
      {
        var convolvedMatrix = new int[outputSize, outputSize];
        for (int t = 0; t < outputSize; t++)
        {
          for (int i = 0; i < outputSize; i++)
          {
            var extractedMatrix = extractMatrix(t * stride, i * stride);
            var dotProduct = applyFilter(filter, extractedMatrix);
            convolvedMatrix[t, i] = dotProduct;
          }
        }
        convolvedMatrices.Add(convolvedMatrix);
      }
    }

    int[,] extractMatrix(int startRow, int startColumn)
    {
      var extractedMatrix = new int[filterSize, filterSize];
      for (int t = 0; t < filterSize; t++)
      {
        for (int i = 0; i < filterSize; i++)
        {
          extractedMatrix[t, i] = paddedInput[startRow + t, startColumn + i];
        }
      }
      return extractedMatrix;
    }

    int applyFilter(int[,] mat1, int[,] mat2)
    {
      var dimension = mat1.GetLength(0);
      var sum = 0;
      for (int i = 0; i < dimension; i++)
      {
        for (int t = 0; t < dimension; t++)
        {
          sum += mat1[i, t] * mat2[i, t];
        }
      }
      return sum;
    }

    int[,] addMatrices(List<int[,]> matrices)
    {
      var dimension = matrices[0].GetLength(0);
      var sumMatrix = new int[dimension, dimension];
      matrices.ForEach((convMatrix) =>
      {
        sumMatrix = addMatrix(convMatrix, sumMatrix);
      });
      return sumMatrix;
    }

    int[,] addMatrix(int[,] mat1, int[,] mat2)
    {
      var dimension = mat1.GetLength(0);
      var sumMatrix = new int[dimension, dimension];
      for (int i = 0; i < dimension; i++)
      {
        for (int t = 0; t < dimension; t++)
        {
          sumMatrix[i, t] = mat1[i, t] + mat2[i, t];
        }
      }
      return sumMatrix;
    }

    public void print()
    {
      var dimension = withBias.GetLength(0);
      for (int i = 0; i < dimension; i++)
      {
        Console.WriteLine();
        for (int t = 0; t < dimension; t++)
        {
          Console.Write(withBias[i, t] + " ");
        }
      }
    }
  }
}
