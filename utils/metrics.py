from sklearn.metrics import mean_absolute_error,r2_score, mean_squared_error
import torch

def predict_and_evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for wind_history, weather_future, future_wind_power in data_loader:
            wind_history, weather_future, future_wind_power = (
                wind_history.to(device),
                weather_future.to(device),
                future_wind_power.to(device)
            )
            output = model(wind_history, weather_future).squeeze()
            predictions.extend(output.cpu().numpy())
            actuals.extend(future_wind_power.cpu().numpy())

    # 计算统计指标
    r2 = r2_score(actuals, predictions)
    rmse = mean_squared_error(actuals, predictions, squared=False)
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)

    return predictions, actuals, r2, rmse, mse, mae