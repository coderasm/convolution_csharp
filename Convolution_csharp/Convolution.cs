using System;
using System.Collections.Generic;
using System.Text;

namespace Convolution_csharp
{
  class Convolution
  {
    //only accepting square input and square filter
    int[,] inputOne = new int[5, 5] { { 0, 1, 0, 2, 1 }, { 2, 2, 2, 0, 1 }, { 0, 1, 2, 1, 2 }, { 2, 1, 1, 1, 0 }, { 2, 1, 2, 1, 0 } };
    int[,] inputTwo = new int[5, 5] { { 0, 0, 1, 1, 0 }, { 0, 1, 0, 0, 0 }, { 2, 2, 1, 1, 2 }, { 1, 1, 2, 2, 0 }, { 1, 0, 0, 2, 1 } };
    int[,] inputThree = new int[5, 5] { { 0, 1, 0, 2, 0 }, { 2, 1, 0, 1, 2 }, { 1, 0, 2, 2, 2 }, { 2, 0, 1, 2, 2 }, { 1, 1, 1, 0, 0 } };
    //int[,] filterOne = new int[3, 3] { { -1, 0, -1 }, { -1, -1, -1 }, { 1, 0, -1 } };
    //int[,] filterTwo = new int[3, 3] { { 0, -1, 0 }, { -1, 0, 0 }, { 1, 0, 1 } };
    //int[,] filterThree = new int[3, 3] { { -1, 0, 1 }, { -1, 1, -1 }, { 1, 0, 1 } };
    //int[,] bias = new int[3, 3] { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };
    int[,] filterOne = new int[3, 3] { { 1, -1, 1 }, { -1, 0, 1 }, { 1, 1, 1 } };
    int[,] filterTwo = new int[3, 3] { { -1, -1, 1 }, { 0, 0, 0 }, { -1, 0, 0 } };
    int[,] filterThree = new int[3, 3] { { -1, 0, 1 }, { 1, 1, 0 }, { 1, 1, 1 } };
    int[,] bias = new int[3, 3] { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0, } };
    List<int[,]> filters;
    List<int[,]> inputs;
    List<int[,]> paddedInputs;
    List<int[,]> convolvedMatrices = new List<int[,]>();
    int inputSize = 0;
    int filterSize = 0;
    int outputSize = 0;
    int stride = 2;
    int zeroPadding = 0;
    int[,] withBias;

    public Convolution()
    {
      filters = new List<int[,]> { filterOne, filterTwo, filterThree };
      inputs = new List<int[,]> { inputOne, inputTwo, inputThree };
      paddedInputs = new List<int[,]> { inputOne, inputTwo, inputThree };
      inputSize = inputOne.GetLength(0);
      filterSize = filterOne.GetLength(0);
      //we are requiring output size to equal filter size
      outputSize = filterSize;
      withBias = new int[outputSize, outputSize];
    }

    public Convolution convolve()
    {
      setStrideAndPadding();
      padInputs();
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

    void padInputs()
    {
      if (zeroPadding > 0)
      {
        var paddedDim = inputSize + 2 * zeroPadding;
        for (int p = 0; p < inputs.Count; p++)
        {
          var input = inputs[p];
          var paddedInput = new int[paddedDim, paddedDim];
          //copy in original input
          for (int i = 0; i < inputSize; i++)
          {
            for (int t = 0; t < inputSize; t++)
            {
              paddedInput[zeroPadding + i, zeroPadding + t] = input[i, t];
            }
          }
          paddedInputs[p] = paddedInput;
        };
      }
    }

    void createConvolvedMatrices()
    {
      //Perform the convolutions
      //loop over filters
      convolvedMatrices = new List<int[,]>();
      for (int p = 0; p < filters.Count; p++)
      {
        var filter = filters[p];
        var paddedInput = paddedInputs[p];
        var convolvedMatrix = new int[outputSize, outputSize];
        for (int t = 0; t < outputSize; t++)
        {
          for (int i = 0; i < outputSize; i++)
          {
            var extractedMatrix = extractMatrix(t * stride, i * stride, paddedInput);
            var modifiedDot = applyFilter(filter, extractedMatrix);
            convolvedMatrix[t, i] = modifiedDot;
          }
        }
        convolvedMatrices.Add(convolvedMatrix);
      }
    }

    int[,] extractMatrix(int startRow, int startColumn, int[,] paddedInput)
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
