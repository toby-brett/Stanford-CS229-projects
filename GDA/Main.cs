using MathNet.Numerics.LinearAlgebra;
using Accord.Math; // matrixes

class MultiNormal 
{
    public static double PDF(Vector<double> mu, Matrix<double> covarMat, Vector<double> X)
    {
        
        int dims = mu.Count;
        double denominator = Math.Pow((2 * Math.PI), (dims / 2)) * Math.Pow(covarMat.Determinant(), 0.5);
        double exponent = -0.5 * ((X - mu) * (covarMat.Determinant())).DotProduct(X - mu); // ToColumnMatrix is the same as transposing
        return (1 / denominator) * Math.Exp(exponent);
    }
}

class Test_Train_Split
{
    
    public static List<string[]> ReadCsvFile(string filepath) // static - instance not needed, List<string[]> returns a list of string arrays where each array is a row of the csv
    {
        List<string[]> rows = new List<string[]>(); // creates empty list, with each row being an array of strings. List<data type of each entry>

        try // if an error occurs it will be caught by the catch block
        {
            string[] lines = File.ReadAllLines(filepath); // converts into an array of strings

            foreach (string line in lines) // iterates through each line of the string array
            {
                string[] values = line.Split(','); // entries in csv are separated by commas

                rows.Add(values); // adds the row to the list
            }
        }
        catch (Exception ex) // ex is the expetion
        {
            Console.WriteLine($"An error occured: {ex.Message}");
        }
        return rows;
    }

    
    public static List<T> Shuffle<T>(List<T> list) // <T> means any datatype
    {
        Random rand = new Random();
        int n = list.Count;
 
        for (int i = n - 1; i > 0; i--) // countus back from end value
        {
            
            int j = rand.Next(0, i + 1); // generates random number
            
            T temp = list[i]; // stores replaced value
            list[i] = list[j]; // switches with random value
            list[j] = temp;
        }

        return list;
    }
    
    
    public static (List<Vector<double>>, List<Vector<double>>, List<int>, List<int>) GetSplit(double percent_train)
    {
        List<string[]> Data = ReadCsvFile("data.csv");

        Data.RemoveRange(0, 51); // first class
        Data = Shuffle(Data);
        
        List<double[]> X = new List<double[]>();
        List<int> y = new List<int>();
        
        foreach (var row in Data) 
        {
            string[] rangeArrayX = new string[4];
            string yValue = row[5];
            
            Array.Copy(row, 1, rangeArrayX, 0, 4); // copies from row, starting at index 1, into range array index 0, copying 4 items
            double[] doubleArray = rangeArrayX.Select(double.Parse).ToArray();
            
            if (yValue.Equals("Iris-versicolor"))
            {
                y.Add(1); 
            }
            else
            {
                y.Add(0); 
            }

            X.Add(doubleArray);
        }

        int index = (int)Math.Round(Data.Count * percent_train);
            
        List<double[]> X_train_list = X.GetRange(0,index); // [:100] python equiv
        List<double[]> X_test_list = X.GetRange(index, Data.Count - index);

        List<Vector<double>> X_train = new List<Vector<double>>();
        List<Vector<double>> X_test = new List<Vector<double>>();
        
        for (int i = 0; i < X_train_list.Count; i++)
        {
            X_train.Add(Vector<double>.Build.DenseOfArray(X_train_list[i]));
        }
        for (int i = 0; i < X_test_list.Count; i++)
        {
            X_test.Add(Vector<double>.Build.DenseOfArray(X_test_list[i]));
        }
        
        List<int> y_train = y.GetRange(0, index);
        List<int> y_test = y.GetRange(index, Data.Count - index);

        return (X_train, X_test, y_train, y_test);
    }
}

class GDA
{
    private (List<Vector<double>>, List<Vector<double>>, List<int>, List<int>) _data;
    
    private List<Vector<double>> X_train;
    private double[] X_test;
    private List<int> y_train;
    private int y_test;
    private double phi;
    private Vector<double> mu0, mu1;
    private Matrix<double> covarMat;
    
    public static (double, Vector<double>, Vector<double>, Matrix<double>) DefineParams(List<Vector<double>> X_train, List<int> y_train)
    {
        double phi = 0D;
        var mu0 = Vector<double>.Build.Dense(4); // 4 features so a 4 dimensional mean
        var mu1 = Vector<double>.Build.Dense(4);
        var covarMat = Matrix<double>.Build.Dense(4, 4);

        int count1s = 0;
        int count0s = 0;
        
        var zippedList = X_train.Zip(y_train, (x, y) => new { x, y }).ToList();

        foreach (var example in zippedList)
        {
            if (example.y == 1)
            {
                count1s += 1;
                phi += 1f; // adds 1 to phi, because phi is simply the proportion of examples which are 1s
                mu1 += example.x; // will divide later for now just sum
            }

            if (example.y == 0)
            {
                count0s += 1;
                mu0 += example.x; // will divide later for now just sum
            }
        }

        mu0 /= count0s;
        mu1 /= count1s;
        phi /= X_train.Count; // averages the phi

        foreach (var example2 in zippedList)
        {
            if (example2.y == 1)
            {
                covarMat += (example2.x - mu1).OuterProduct((example2.x - mu1)); //sum first
            }
            else
            {
                covarMat += (example2.x - mu0).OuterProduct((example2.x - mu0)); //sum first
            }
        }

        covarMat /= X_train.Count;
        return (phi, mu0, mu1, covarMat); // all works i think
    }

    
    public GDA()
    {
        _data = Test_Train_Split.GetSplit(0.8f);
        
        X_train = _data.Item1; 
        y_train = _data.Item3;
        
        var parameters = DefineParams(X_train, y_train); // X_train y_train
        phi = parameters.Item1;
        mu0 = parameters.Item2;
        mu1 = parameters.Item3;
        covarMat = parameters.Item4;
        
    }
    public int Test(int i) // index to test 
    {
        int y_test = _data.Item4[i];
        var X_test = _data.Item2[i]; // only test one index at a time
        
        double pdf0 = MultiNormal.PDF(mu0, covarMat, X_test); // X_test 
        double pdf1 = MultiNormal.PDF(mu1, covarMat, X_test); // same idem for a different distribution

        if (pdf1 > pdf0 && y_test == 1)
        {
            return 1; // correct
        }

        if (pdf1 < pdf0 && y_test == 0)
        {
            return 1;
        }
        return 0; // wrong
    }
}

class main
{
    static void Main() // main method, the entry point of the program
    {
        string csvFilePath = "data.csv";

        List<string[]> csvData = Test_Train_Split.ReadCsvFile(csvFilePath);
        GDA model = new GDA();

        int correct = 0;
        for (int i = 0; i < 20; i++) // 20 test examles
        {
            if (model.Test(i) == 1)
            {   
                correct += 1;
            }
        }
        Console.WriteLine("accuracy = " + ((double)correct / 20).ToString());
    }
}
