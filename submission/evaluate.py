import numpy as np
import torch
import cv2
from climatehack import BaseEvaluator
from model2 import ConvNextUnet

def compute_flows(images, **kwargs):
    flows = []
    for image_i in range(images.shape[0] - 1):
        flow = cv2.calcOpticalFlowFarneback(
            prev=images[image_i], next=images[image_i+1], flow=None, **kwargs)
        flows.append(flow)
    return np.average(np.stack(flows), axis=0).astype(np.float32)

def remap_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Takes an image and warps it forwards in time according to the flow field.
    
    Args:
        image: The grayscale image to warp.
        flow: A 3D array.  The first two dimensions must be the same size as the first two
            dimensions of the image.  The third dimension represented the x and y displacement.
            
    Returns:  Warped image.
    """
    # Adapted from https://github.com/opencv/opencv/issues/11068
    height, width = flow.shape[:2]
    remap = -flow.copy()
    remap[..., 0] += np.arange(width)  # x map
    remap[..., 1] += np.arange(height)[:, np.newaxis]  # y map
    return cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        self.model = ConvNextUnet(
            dim=64,
            out_dim=24,
            channels=12,
            with_time_emb=False
        )
        self.model.load_state_dict(torch.load("model4.pth"))
        self.model.eval()


        self.model2 = ConvNextUnet(
            dim=64,
            out_dim=24,
            channels=12,
            with_time_emb=False
        )
        self.model2.load_state_dict(torch.load("model3.pth"))
        self.model2.eval()

        # self.model2 = Unet()
        # self.model2.load_state_dict(torch.load("unet.pth"))
        # self.model2.eval()

        # self.refine = ConvNextUnet(
        #     dim=64,
        #     out_dim=24,
        #     channels=36,
        #     with_time_emb=False,
        #     aux_cls=True
        # )
        # self.refine.load_state_dict(torch.load("refine.pth"))
        # self.refine.eval()

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        # assert coordinates.shape == (2, 128, 128)
        # assert data.shape == (12, 128, 128)

        # flow = compute_flows(data, pyr_scale=0.5, levels=3, winsize=15, 
        #                         iterations=10, poly_n=5, poly_sigma=1.2, 
        #                         flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # predictions = []
        # for i in range(24):
        #     predictions.append(remap_image(data[-1], flow*(i+1)))         
        # predictions = np.stack(predictions, axis=0)
        # predictions = predictions[:, 32:96, 32:96]
        # assert predictions.shape == (24, 64, 64)
        # return predictions

        with torch.no_grad():
            input = torch.from_numpy(data) / 1024
            input = input.view(1, 12, 128, 128)

            prediction_1, _ = self.model(input, 0)
            prediction_2, _ = self.model(prediction_1[:, :12], 0)
            prediction_3 = torch.cat((prediction_1[:, :12], prediction_2[:, :12]), dim=1)

            prediction = prediction_1 * 0.6 + prediction_3 * 0.4 #(prediction_1 + prediction_3) / 2

            prediction_1, _ = self.model2(input, 0)
            prediction_2, _ = self.model2(prediction_1[:, :12], 0)
            prediction_3 = torch.cat((prediction_1[:, :12], prediction_2[:, :12]), dim=1)

            prediction2 = prediction_1 * 0.6 + prediction_3 * 0.4 #(prediction_1 + prediction_3) / 2

            prediction = prediction * 0.5 + prediction2 * 0.5

            prediction = prediction.view(24, 128, 128).detach().numpy()
            prediction = prediction[:, 32:96, 32:96]*1024
            prediction = np.clip(prediction, 0, 1024)

            # prediction = np.rint(prediction)

            assert prediction.shape == (24, 64, 64)
            return prediction
            
        # last_img = data[-1, 32:96, 32:96]
        # return np.tile(last_img, (24, 1, 1))

def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
