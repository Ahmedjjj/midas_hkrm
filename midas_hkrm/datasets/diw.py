import os

from midas_hkrm.datasets import ZeroShotDataset


class DIW(ZeroShotDataset):
    """
    DIW Test set: http://www-personal.umich.edu/~wfchen/depth-in-the-wild/
    Ground truth: ordinal relations
    """

    def __init__(self):
        super().__init__(test=True)

    @property
    def name(self):
        return "DIW"

    @staticmethod
    def __parse_annotation(line: str):
        """Parses a line from the DIW annotation file
        From what we understand (we could not find documentation on this), the line has the format:
        point1_x, point1_y, point2_x, point2_y, > OR <, image_height, img_width

        The sign encodes the relationship (depth space) between point1 and point2
        Args:
            line (str): One annotation line

        Returns:
            Tuple: x_further, y_further, x_closer, y_closer
        """
        x1, y1, x2, y2, sign, _, _ = line.split(",")
        if sign == ">":
            return int(x1) - 1, int(y1) - 1, int(x2) - 1, int(y2) - 1
        else:
            return int(x2) - 1, int(y2) - 1, int(x1) - 1, int(y1) - 1

    @property
    def locations(self):
        return []

    @property
    def all_samples(self):
        all_samples = []
        annotation_filepath = os.path.join(
            self.dataset_path, "DIW_Annotations", "DIW_test.csv"
        )
        with open(annotation_filepath) as f:
            lines = f.readlines()
            files = [s.rstrip() for s in lines[::2]]
            annotations = [s.rstrip() for s in lines[1::2]]

        all_samples = list(
            zip(
                [os.path.join(self.dataset_path, f) for f in files],
                [self.__parse_annotation(a) for a in annotations],
            )
        )

        return all_samples

    def __getitem__(self, index):
        item = self._map[index]
        img_path, annotation = item
        return self.get_image(img_path), annotation
