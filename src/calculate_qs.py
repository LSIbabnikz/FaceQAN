
import sys
sys.path.append("./")

import argparse
import os
import pickle

from PIL import Image
import torch
from torchvision.transforms import transforms

from util import F, load_cosface, add_norm_to_model
from bim_attack import basic_iterative_method as BIM
from symmetry import symmetry_estimation


def __calculate_score_single__(model: torch.nn.Module, image_transform: transforms.Compose, image_path: str, eps: float, l: int, k: int, p: int) -> float:
    """Method performs FaceQAN over individual images

    Args:
        model (torch.nn.Module): FR model used for adversarial attack.
        image_transform (transforms.Compose): Image transformation, without normalization.
        image_path (str): Path to image, for which the quality score will be predicted.
        eps (float): ε-parameter from the paper.
        l (int): l-parameter from the paper, controling the number of FGSM iterations to perform.
        k (int): k-parameter from the paper, controling the number of adversarial examples generated.
        p (int): p-parameter from the paper, used in the final quality score calculation as the exponent.

    Returns:
        float: The calculated quality score 
    """

    assert os.path.exists(image_path), f" Image path {image_path} does not exist! "

    print(f" => Extracting quality for {image_path}")

    image = Image.open(image_path).convert("RGB")

    S_i = BIM(model, image_transform(image), eps=eps, iter=l)

    q_s = symmetry_estimation(model, image_transform, image)

    quality_score = F(S_i, q_s)

    return quality_score


def __calculate_score_batch__(image_paths: list, eps: float, l: int, k: int, p: int) -> dict:
    """Performs all the steps of the FaceQAN approach based on the input parameters

    Args:
        image_path (str): Path to the image for which we want to generate the quality score.
        eps (float): ε-parameter from the paper, controlling both the amount of adversarial noise as well as the uniform distribution.
        l (int): l-parameter from the paper, controlling the number of FGSM iterations to perform.
        k (int): k-parameter from the paper, controlling the number of adversarial examples generated.
        p (int): p-parameter from the paper, used in the final quality score calculation as the exponent.

    Returns:
        dict: Dictionary where key=image_path, value=generated quality score
    """

    """
        In case you wish to use a different FR model simply replace the load_cosface function with a custom function 
        that loads your desired FR model, additionally change the mean, st.deviation and transform used by your custom FR model
    """
    model : torch.nn.Module = load_cosface().eval().cuda()
    mean, std = [.5, .5, .5], [.5, .5, .5]
    image_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    model = add_norm_to_model(model, mean=mean, std=std)

    quality_scores = dict(map(lambda image_path: (image_path, __calculate_score_single__(model, image_transforms, image_path, eps, l, k, p)), image_paths))

    return quality_scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_paths", required=True, type=str, nargs="+", help=" Location of images for which to generate scores ")
    parser.add_argument("-e", "--epsilon", type=float, default=0.001, help=" ε controls the uniform distribution spread, as well as the amount of adversarial noise added using BIM. ")
    parser.add_argument("-l", "--bim_iterations", type=int, default=5, help=" l controls the number of BIM iterations. ")
    parser.add_argument("-k", "--batch_size", type=int, default=10, help=" k controls the number of adversarial examples generated in each iteration. ")
    parser.add_argument("-p", "--exponent", type=int, default=5, help=" p controls the exponent used in the final calculation of the quality score. ")
    parser.add_argument("-sp", "--save_path", required=True, type=str, default=".", help=" Path to location where the results will be stored. ")
    args = parser.parse_args()

    assert args.epsilon > 0. and args.epsilon < 1., f" Parameter ε should be in range (0., 1.) "
    assert args.bim_iterations >= 1, f" Number of iterations should be positive "
    assert args.batch_size > 1, f" Batch size should be larger than 1 "
    assert args.exponent >= 1, f" Exponent should be positive "
    assert os.path.exists(args.save_path), f" Given save path {args.save_path} does not exist "

    print(f"{''.join(['#' for _ in range(100)])}\
            {os.linesep}=> Running FaceQAN over {len(args.image_paths)} images, using the following configuration:\
            {os.linesep}\t -> ε = {args.epsilon}\
            {os.linesep}\t -> l = {args.bim_iterations}\
            {os.linesep}\t -> k = {args.batch_size}\
            {os.linesep}\t -> p = {args.exponent}\
            {os.linesep}{''.join(['#' for _ in range(100)])}")
    
    qs = __calculate_score_batch__(args.image_paths, args.epsilon, args.bim_iterations, args.batch_size, args.exponent)

    print(f"{''.join(['#' for _ in range(100)])}\
            {os.linesep} => Saving results to {os.path.join(args.save_path, 'results.pkl')}\
            {os.linesep}{''.join(['#' for _ in range(100)])}")

    with open(os.path.join(args.save_path, "results.pkl"), "wb") as pkl_out:
        pickle.dump(qs, pkl_out)
