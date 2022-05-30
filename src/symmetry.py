
import torch
from torchvision.transforms import transforms
from PIL import Image


def symmetry_estimation(model: torch.nn.Module, image_transform: transforms.Compose, image: Image.Image) -> float:
    """Perfroms the symmetry estimation step from the FaceQAN paper, estimating the effect of head pose on the final quality score

    Args:
        model (torch.nn.Module): Given FR model.
        image_transform (transforms.Compose): Image transforms, without normalization.
        image (PIL.Image): Input image, for which to generate quality score.

    Returns:
        float: Symmetry score of given image.
    """

    cos_sim = torch.nn.CosineSimilarity(dim=1)

    image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)

    embeds = model(torch.vstack((image_transform(image).unsqueeze(0), image_transform(image_flip).unsqueeze(0))).cuda()).detach()

    return cos_sim(embeds[0].unsqueeze(0), embeds[1].unsqueeze(0)).item()