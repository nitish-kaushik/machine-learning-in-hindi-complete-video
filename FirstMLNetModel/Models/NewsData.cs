using Microsoft.ML.Data;

namespace FirstMLNetModel.Models;

public class NewsData
{
    // [NoColumn]
    // public int Id { get; set; }
    [LoadColumn(1)]
    public string Content { get; set; } = null!;
    [LoadColumn(2)]
    public string Category { get; set; } = null!;
}
