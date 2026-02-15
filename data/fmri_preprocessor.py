import numpy as np
import pandas as pd
from nilearn import image
from nilearn.input_data import NiftiMasker
from tqdm import tqdm


class FMRI_Preprocessor:
    """
    fMRI数据预处理器
    
    使用NiftiMasker进行fMRI数据的标准化、去趋势、滤波和平滑处理。
    
    Attributes:
        masker: NiftiMasker实例，用于数据预处理
        ts_: 预处理后的时间序列数据 (numpy array)
        fmri_img_: 原始fMRI图像
        clean_img: 预处理后的清洁fMRI图像 (NIfTI)
        confounds_: 混杂变量数据
    """
    def __init__(self, tr=2.0, high_pass=0.01, low_pass=0.1, smoothing_fwhm=6.0):
        self.masker = NiftiMasker(
            standardize=True,
            detrend=True,
            t_r=tr,
            high_pass=high_pass,
            low_pass=low_pass,
            smoothing_fwhm=smoothing_fwhm
        )
        self.ts_ = None
        self.fmri_img_ = None
        self.confounds_ = None
        self.tr=tr

    def preprocess(self, fmri_file, confounds_file=None):
        """
        预处理fMRI数据
        
        Args:
            fmri_file: fMRI NIfTI文件路径或NIfTI图像对象
            confounds_file: 混杂变量文件路径 (.tsv或.csv)
            
        Returns:
            nibabel.Nifti1Image: 预处理后的清洁fMRI图像
            
        Note:
            预处理后的时间序列数据可通过get_time_series()方法获取
        """
        steps = [
            "读取 fMRI NIfTI 文件",
            "读取混杂变量文件" if confounds_file is not None else "跳过混杂变量",
            "应用 NiftiMasker 进行预处理"
        ]
        with tqdm(total=len(steps), desc="fMRI Preprocessing", ncols=100) as pbar:
            # Step 1
            self.fmri_img_ = image.load_img(fmri_file)
            pbar.set_postfix_str("fMRI 加载完成")
            pbar.update(1)

            # Step 2
            confounds = None
            if confounds_file is not None:
                if confounds_file.endswith(".tsv"):
                    confounds = pd.read_csv(confounds_file, sep="\t")
                elif confounds_file.endswith(".csv"):
                    confounds = pd.read_csv(confounds_file)
                else:
                    raise ValueError("Confounds file must be .tsv or .csv")
                confounds = confounds.fillna(0)
                self.confounds_ = confounds
                pbar.set_postfix_str("混杂变量加载完成")
            else:
                pbar.set_postfix_str("未提供混杂变量")
            pbar.update(1)

            # Step 3
            ts = self.masker.fit_transform(self.fmri_img_, confounds=confounds)
            self.ts_ = ts.astype(np.float32)
            
            # 关键修复：生成 clean_img
            self.clean_img = self.inverse_transform(self.ts_)
            
            pbar.set_postfix_str("预处理完成 + clean_img 生成")
            pbar.update(1)
        
        # Return both clean_img (NIfTI) and time series (numpy array)
        return self.clean_img
        
    def get_time_series(self):
        if self.ts_ is None:
            raise ValueError("请先调用 preprocess() 进行预处理")
        return self.ts_

    def inverse_transform(self, ts):
        with tqdm(total=1, desc="🔄 Inverse Transform", ncols=100) as pbar:
            img = self.masker.inverse_transform(ts)
            pbar.set_postfix_str("NIfTI 生成完成")
            pbar.update(1)
        return img

    def save_clean_img(self, save_path):
        if self.ts_ is None:
            raise ValueError("请先调用 preprocess()")

        with tqdm(total=2, desc="💾 Save Clean fMRI", ncols=100) as pbar:
            clean_img = self.inverse_transform(self.ts_)
            pbar.set_postfix_str("NIfTI 转换完成")
            pbar.update(1)

            clean_img.to_filename(save_path)
            pbar.set_postfix_str("保存完成")
            pbar.update(1)

        print(f"🎉 Preprocessed fMRI image saved to {save_path}")
