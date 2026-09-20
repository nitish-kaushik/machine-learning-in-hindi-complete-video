using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});

builder.Services.AddSingleton(sp =>
{
    var mlContext = new Microsoft.ML.MLContext();
    var modelPath = Path.Combine(AppContext.BaseDirectory, "NewsModel.zip");
    var loadedModel = mlContext.Model.Load(modelPath, out var modelSchema);
    return mlContext.Model.CreatePredictionEngine<News, NewsPrediction>(loadedModel);
});

builder.Services.AddOpenApi();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseCors();
app.UseHttpsRedirection();

app.MapPost("/ask", ([FromBody] News question,
        [FromServices] Microsoft.ML.PredictionEngine<News, NewsPrediction> predictionEngine) =>
{
    var answer = new NewsPrediction()
    {
        PredictedLabel = predictionEngine.Predict(question).PredictedLabel
    };

    return Results.Ok(answer);
})
.WithName("getAnswer");

app.Run();
