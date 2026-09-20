// See https://aka.ms/new-console-template for more information

using FirstMLNetModel.Models;
using Microsoft.ML;

var context = new MLContext();

var dataPath = Path.Combine(AppContext.BaseDirectory, "Data", "news.csv");

var data = context.Data.LoadFromTextFile<NewsData>(dataPath, hasHeader: true, separatorChar: ',');

var dataSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

var trainData = dataSplit.TrainSet;
var testData = dataSplit.TestSet;

var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(NewsData.Content))
    .Append(context.Transforms.Conversion.MapValueToKey("Label", nameof(NewsData.Category)))
    .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
    .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

var model = pipeline.Fit(trainData);

var predictions = model.Transform(testData);

var metrics = context.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");
Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:P2}");

var modelPath = Path.Combine(AppContext.BaseDirectory, "NewsModel.zip");
context.Model.Save(model, trainData.Schema, modelPath);
Console.WriteLine($"Model saved to: {modelPath}");

// var predictionEngine = context.Model.CreatePredictionEngine<NewsData, NewsPrediction>(model);
//
// while (true)
// {
//     Console.WriteLine("Enter news content (or 'exit' to quit):");
//     var content = Console.ReadLine();
//     if (string.Equals(content, "exit", StringComparison.OrdinalIgnoreCase))
//     {
//         break;
//     }
//
//     var sampleNews = new NewsData
//     {
//         Content = content,
//     };
//
//     var prediction = predictionEngine.Predict(sampleNews);
//     Console.WriteLine($"Predicted Category: {prediction.PredictedLabel}");
// }
