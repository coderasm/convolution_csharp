using System;

namespace Convolution_csharp
{
  class Program
  {
    static void Main(string[] args)
    {
      var convolution = new Convolution();
      convolution.convolve().print();
    }
  }
}
