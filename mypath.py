class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/path/to/ucf101/video'

            # Save preprocess data into output_dir
            output_dir = '/path/to/ucf101_custom'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
