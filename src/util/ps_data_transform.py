import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Any, Dict
import numpy as np

class ConvertDtype(A.ImageOnlyTransform):
    """Convert float64 to float32 to avoid albumentations dtype issues"""
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img.astype(np.float32)


class PatchwiseNormalize(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # img shape: (H, W, C)
        mean = img.mean(axis=(0, 1), keepdims=True)
        std = img.std(axis=(0, 1), keepdims=True)
        std[std == 0] = 1.0  # Prevent division by zero
        return (img - mean) / std

class MixedNormalize(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.astype(np.float32)
        out = img.copy()
        # Normalize first 3 channels with ImageNet stats
        if img.shape[2] >= 3:
            out[..., :3] = (img[..., :3] - self.imagenet_mean) / self.imagenet_std
        # Patch-wise normalize remaining channels
        if img.shape[2] > 3:
            patch = img[..., 3:]
            mean = patch.mean(axis=(0, 1), keepdims=True)
            std = patch.std(axis=(0, 1), keepdims=True)
            std[std == 0] = 1.0
            out[..., 3:] = (patch - mean) / std
        return out


class MinMaxScale(A.ImageOnlyTransform):
    def __init__(self, min_vals, max_vals, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.min_vals = np.array(min_vals, dtype=np.float32)
        self.max_vals = np.array(max_vals, dtype=np.float32)

    def apply(self, image, **params):
        # image shape: (H, W, C)
        image = (image - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)
        image = np.clip(image, 0, 1) * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        # image = np.clip(image, 0, 1)
        return image

class RandomShiftAugmentation(A.DualTransform):
    """
    对CHM标签进行随机空间偏移增强
    
    目的：
    - 模拟CHM标签与遥感输入之间的对齐误差
    - 增加训练数据多样性
    - 使模型更鲁棒地处理位置偏差
    
    原理：
    - 在x和y方向随机偏移标签
    - 使用双线性插值填充边界
    - 保持输入（image）不变，只偏移输出（mask）
    - 这样模型学到：给定某个输入，如何处理各种可能的偏移标签
    """
    
    def __init__(self, shift_limit=3, always_apply=False, p=0.5):
        """
        Args:
            shift_limit: 最大偏移像素数
                        如果 shift_limit=3，则 shift_x, shift_y ∈ [-3, 3]
            always_apply: 总是应用此变换
            p: 应用概率
        """
        super().__init__(always_apply, p)
        self.shift_limit = shift_limit
    
    def apply(self, img, **params):
        """保持输入图像不变"""
        return img
    
    def apply_to_mask(self, mask, **params):
        """对标签应用随机偏移"""
        # 随机选择偏移量
        shift_x = np.random.randint(-self.shift_limit, self.shift_limit + 1)
        shift_y = np.random.randint(-self.shift_limit, self.shift_limit + 1)
        
        # 应用偏移
        shifted_mask = self._shift_image(mask, shift_x, shift_y)
        
        return shifted_mask
    
    def _shift_image(self, image, shift_x, shift_y):
        """
        对图像进行平移
        
        Args:
            image: [H, W] 或 [H, W, C]
            shift_x: x方向偏移（列方向）
            shift_y: y方向偏移（行方向）
        
        Returns:
            shifted_image: 平移后的图像
        """
        h, w = image.shape[:2]
        
        # 创建变换矩阵（平移）
        # [1 0 shift_x]
        # [0 1 shift_y]
        M = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])
        
        # 使用仿射变换进行平移
        # borderMode: cv2.BORDER_CONSTANT 用0填充边界
        # borderValue: 填充值（对于CHM数据，可以用0表示无效）
        shifted = cv2.warpAffine(
            image, 
            M, 
            (w, h),
            flags=cv2.INTER_LINEAR,  # 双线性插值
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0  # 边界用0填充（表示无效数据）
        )
        
        return shifted
    
    def get_transform_init_args_names(self):
        return ("shift_limit",)

def get_multiscale_transforms(
    patch_sizes: List[Tuple[int, int]], 
    num_channels: int = 6, 
    is_training: bool = True, 
    is_output: bool = False,
    normalize_target: bool = False,
    minmax_target: bool = False,
    unit_scale_ratio: float = 1,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    use_global_minmax: bool = False,
    use_global_norm: bool = False,
    use_local_norm: bool = False,
    globalnorm_stats: dict = {}, # mean, std, etc.
    targetnorm_stats: dict = {},
    use_shift_augmentation: bool = False, # 是否使用随机偏移增强（仅对标签）
    shift_limit: int = 3, # 偏移范围 [-shift_limit, +shift_limit] 像素
    shift_augmentation_p: float = 0.5 # 使用偏移增强的概率
) -> A.Compose:
    """Get data transformations for multi-scale training"""
    print('num_channels:', num_channels)


    # Create normalization parameters based on number of channels
    # Using ImageNet stats for RGB, extending for more channels
    if use_global_minmax:
        min_max = MinMaxScale(globalnorm_stats["p1"], globalnorm_stats["p99"])

    if use_global_norm:
        norm = A.Normalize(mean=globalnorm_stats['mean'], std=globalnorm_stats['std'])
        print(f"After min-max scaling, using patch-wise normalization for {num_channels} channels")
        print('=' * 100)
        # norm = PatchwiseNormalize()
        print('Using pre-calculated global mean std')
    elif use_local_norm:
        if num_channels == 3:
            print("Using ImageNet stats for RGB")
            print('='*100)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            norm = A.Normalize(mean=mean, std=std)
        else:
            print(f"Using patch-wise normalization for {num_channels} channels")
            print('='*100)
            norm = PatchwiseNormalize()
            # print(f"Using mixed normalization for {num_channels} channels")
            # print('='*100)
            # norm = MixedNormalize()
    else:
        print('Input without any normalization')
        norm = None
    # list of transforms for input data
    transform_list_input = []
    if use_global_minmax:
        transform_list_input.append(min_max)

    if is_training:
        transform_list_input.extend([A.RandomBrightnessContrast(p=0.5),
                                    A.RandomGamma(p=0.5)])

    # apply to all input
    if norm is not None:
        transform_list_input.append(norm)
    transform_list_input.extend([
                                ConvertDtype(),  # Convert float64 to float32 first
                                ToTensorV2()])

    # for output
    scale_target = A.Lambda(image=lambda x, **kwargs: x * 1e+6)

    unit_rescale_target = A.Lambda(image=lambda x, **kwargs: x * unit_scale_ratio)

    if is_training:
        if is_output:
            # if normalize_target is True, use target global mean and std
            if normalize_target:

                output_transforms = []
                if use_shift_augmentation:
                    output_transforms.append(
                        RandomShiftAugmentation(
                            shift_limit=shift_limit,
                            p=shift_augmentation_p
                        )
                    )

                norm_target = A.Normalize(mean=target_mean, std=target_std)

                output_transforms.extend([
                    norm_target,
                    scale_target,
                    ConvertDtype(),
                    ToTensorV2()
                ])
                return A.Compose(output_transforms, is_check_shapes=False)
                
            if minmax_target:
                
                output_transforms = []
                if use_shift_augmentation:
                    output_transforms.append(
                        RandomShiftAugmentation(
                            shift_limit=shift_limit,
                            p=shift_augmentation_p
                        )
                    )

                min_max_target_compose = MinMaxScale(targetnorm_stats["p1"], targetnorm_stats["p99"])

                output_transforms.extend([
                    min_max_target_compose,
                    ConvertDtype(),
                    ToTensorV2()
                ])
                return A.Compose(output_transforms, is_check_shapes=False)
                
            else:

                output_transforms = []
                if use_shift_augmentation:
                    output_transforms.append(
                        RandomShiftAugmentation(
                            shift_limit=shift_limit,
                            p=shift_augmentation_p
                        )
                    )
                output_transforms.extend([
                    unit_rescale_target,
                    ConvertDtype(),
                    ToTensorV2()
                ])
                return A.Compose(output_transforms, is_check_shapes=False)
                
        else:
            return A.Compose(transform_list_input, is_check_shapes=False)
    else:
        if is_output:
            if normalize_target:
                norm_target = A.Normalize(mean=target_mean, std=target_std)
                return A.Compose([
                    norm_target,
                    scale_target,
                    ConvertDtype(),
                    ToTensorV2()
                ], is_check_shapes=False)
            if minmax_target:
                min_max_target_compose = MinMaxScale(targetnorm_stats["p1"], targetnorm_stats["p99"])
                return A.Compose([
                    min_max_target_compose,
                    ConvertDtype(),
                    ToTensorV2()
                ], is_check_shapes=False)            
            else:
                return A.Compose([
                    unit_rescale_target,
                    ConvertDtype(),
                    ToTensorV2()
                ], is_check_shapes=False)
        else:
            return A.Compose(transform_list_input, is_check_shapes=False)

def get_basic_transforms(num_channels: int = 6, is_training: bool = True) -> A.Compose:
    """Get basic data transformations"""
    # Create normalization parameters based on number of channels
    if num_channels == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        norm = A.Normalize(mean=mean, std=std)
    else:
        norm = PatchwiseNormalize()
    
    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            norm,
            ToTensorV2()
        ], is_check_shapes=False)
    else:
        return A.Compose([
            norm,
            ToTensorV2()
        ], is_check_shapes=False)