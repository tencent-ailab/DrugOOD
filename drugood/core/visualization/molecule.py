# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
import numpy as np

# A small value
EPS = 1e-2


def imshow_infos(input):
    """Show mol with extra infomation.

    Args:
        input (str): The smile to be displayed.
        infos (dict): Extra infos to display in the image.
        text_color (:obj:`mmcv.Color`/str/tuple/int/ndarray): Extra infos
            display color. Defaults to 'white'.
        font_size (int): Extra infos display font size. Defaults to 26.
        row_width (int): width between each row of results on the image.
        win_name (str): The image title. Defaults to ''
        show (bool): Whether to show the image. Defaults to True.
        fig_size (tuple): Image show figure size. Defaults to (15, 10).
        wait_time (int): How many seconds to display the image. Defaults to 0.
        out_file (Optional[str]): The filename to write the image.
            Defaults to None.
    Returns:
        np.ndarray: The image with extra infomations.
    """
    NotImplemented
