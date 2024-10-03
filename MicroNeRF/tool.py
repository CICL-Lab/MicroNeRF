
def perceptual_loss(y, y_pred, epsilon=1e-3):
    y_pred_sg = y_pred.detach()
    loss = ((y_pred/255 - y/255) / (y_pred_sg/255 + epsilon))**2
    return loss.mean()
