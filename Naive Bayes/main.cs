// See https://aka.ms/new-console-template for more information

using MathNet.Numerics.LinearAlgebra;
using System.Text.RegularExpressions;

class DataHandler
{
    public static List<string> vocab = GenerateVocab();
    public static List<string> GenerateVocab()
    {
        string filepath = "IMDB Dataset.csv";
        HashSet<string> vocabHash = new HashSet<string>();

        string[] lines = File.ReadAllLines(filepath);
        foreach (string line in lines)
        {
            string sentenceNoPunc = Regex.Replace(line, @"[^\w\s]", " ").ToLower();
            
            string[] words = sentenceNoPunc.Split();
            
            foreach (string word in words)
            {
                vocabHash.Add(word); // doesnt allow dubplicates (hashset)
            }
        }

        List<string> vocab = vocabHash.ToList();
        Console.WriteLine(vocab.Count);
        return vocab;
    }
    
    public static Vector<double> exampleToVector(string sentence)
    {
        string sentenceNoPunc = Regex.Replace(sentence, @"[^\w\s]", " ").ToLower();
        string[] sentenceArray = sentenceNoPunc.Split(' ');
        var SentenceVec = Vector<double>.Build.Dense(101936);
        
        foreach (string word in sentenceArray)
        {
            if (vocab.IndexOf(word) != -1) // so rouge spaces or punc dont matter
            {
                SentenceVec[vocab.IndexOf(word)] = 1;
            }
        }
        return SentenceVec;
    }

    public static (List<Vector<double>>, List<int>) IMDBloader()
    {

        string filepath = "IMDB Dataset.csv";
        var X_train = new List<Vector<double>>();
        var y_train = new List<int>();

        int i = 0;
        
        foreach (var line in File.ReadLines(filepath))
        {
            i += 1;
            if (i == 1) // header
            {
                continue;
            }
            if (i > 20000 )
            {
                break;
            }
            
            Match match = Regex.Match(line, @"\b\w+\b(?=\W*$)"); 
            string sentiment = match.Success ? match.Value : "";
            
            Vector<double> reveiw = exampleToVector(line);
            
            X_train.Add(reveiw);
            
            if (sentiment == "positive")
            {
                y_train.Add(0);
            }
            else
            {
                y_train.Add(1);
            }
        }
        return (X_train, y_train);
    }
}

class Model
{
    public static (double, Vector<double>, Vector<double>) MLE()
    {
        var Data = DataHandler.IMDBloader();
        List<Vector<double>> X_train = Data.Item1;
        List<int> y_train = Data.Item2;
            
        var zippedList = X_train.Zip(y_train, (X, y) => new { X, y }).ToList();

        double count0 = 2; // laplace smoothing, assume a examples of 0s and an example of 1s for each class
        double count1 = 2;
        
        double phi = 0; // proportion of examples which are negative
        Vector<double> phi0 = Vector<double>.Build.Dense(101936); // each word in the dict, phi 0 means positive
        Vector<double> phi1 = Vector<double>.Build.Dense(101936); // negative

        for (int j = 0; j < 101936; j++) // Laplace Smoothing
        {
            phi0[j] = 1;
            phi1[j] = 1; 
        }
        
        foreach (var i in zippedList)
        {
            if (i.y == 1)
            {
                count1 += 1;
                phi1 += i.X; // as i.X is already a multi hot vector with the same encoding as phi1, just add them #optimizingKing
            }

            if (i.y == 0)
            {
                count0 += 1;
                phi0 += i.X; // sum for now, divide after
            }
        }
        phi1 /= count1;
        phi0 /= count0;
        phi = count1 / (count1 + count0);

        return (phi, phi0, phi1);
    }

    public static float GetP(Vector<double> X, double phi, Vector<double> phi0, Vector<double> phi1)
    {
        double Pgiven1 = 1;
        double Pgiven0 = 1;
        
        for (int j = 0; j < 101936; j++)
        {
            if (X[j] == 1) // is this word present
            {
                double PofXjAppearingGiven0 = phi0[j];
                double PofXjAppearingGiven1 = phi1[j]; // chance of this word appearing
                
                Pgiven1 *= PofXjAppearingGiven1;
                Pgiven0 *= PofXjAppearingGiven0;
            }
        }

        if (Pgiven1 < Pgiven0)
        {
            return 0;
        }

        return 1;
    }

    public static float Test(String input, double phi, Vector<double> phi0, Vector<double> phi1)
    {
        Vector<double> Vec = DataHandler.exampleToVector(input);
        return GetP(Vec, phi, phi0, phi1);
    }
}

class main
{
    public static void Main()
    {
        var parameters = Model.MLE();
        var phi = parameters.Item1;
        var phi0 = parameters.Item2;
        var phi1 = parameters.Item3;
        Console.WriteLine("done");
        while (true)
        {
            string userInput = Console.ReadLine();
            
            if (Model.Test(userInput, phi, phi0, phi1) == 1)
            {
                Console.WriteLine("Negative");
            }
            else
            {
                Console.WriteLine("Positive");
            }
        }
    }
}

