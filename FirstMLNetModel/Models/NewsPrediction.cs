namespace FirstMLNetModel.Models;

public class NewsPrediction
{
    public string PredictedLabel { get; set; } = null!;
    public float[] Score { get; set; } = null!;
}
