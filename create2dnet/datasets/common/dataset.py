import os
import zipfile
import wget


class Dataset(object):

    def __init__(self, name, url, path='orig_data'):
        try:
            os.makedirs(path)
        except OSError:
            pass
        self.name = name
        self.url = url
        self.path = path
        self.joints = None

    def load(self):

        self._download()
        self.joints = self._load_joints()

    def get_data(self, i):

        label = self._get_data_label(i)
        joint = self.joints[i]
        image_file, image = self._get_image(i)
        if image is None:
            raise FileNotFoundError('{0} is not found.'.format(image_file))
        return label, joint, image_file, image

    def __len__(self):
        return len(self.joints)

    def _download(self):

        path = os.path.join(self.path, self.name)
        if not os.path.isdir(path):
            path = wget.download(self.url, self.path)
            with zipfile.ZipFile(path, 'r') as zip_file:
                zip_file.extractall(self._get_extract_path())
            os.remove(path)

    def _get_extract_path(self):
        raise NotImplementedError

    def _load_joints(self):
        raise NotImplementedError

    def _get_image(self, i):
        raise NotImplementedError

    def _get_data_label(self, i):
        raise NotImplementedError
