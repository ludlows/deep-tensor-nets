import logging
import tensorflow as tf
import numpy as np

from utils import myshape


def inter_atomic_distances(positions, cell, pbc, cutoff):
    '''

    '''
    with tf.variable_scope('distance'):
        # calculate heights
        icell = tf.matrix_inverse(cell)
        # account for zero cell in case of no pbc
        c = tf.reduce_sum(tf.cast(pbc, tf.int32)) > 0
        icell = tf.cond(c, lambda: tf.matrix_inverse(cell),
            lambda:tf.eye(3))

        height = 1.0 / tf.sqrt(tf.reduce_sum(tf.square(icell), 0))
        extent = tf.where(tf.cast(pbc, tf.bool),
                          tf.cast(tf.floor(cutoff / height), tf.int32),
                          tf.cast(tf.zeros_like(height), tf.int32))
        n_reps = tf.reduce_prod(2 * extent + 1)

        # replicate atoms
        r = tf.range(-extent[0], extent[0] + 1)
        v0 = tf.expand_dims(r, 1)
        v0 = tf.tile(v0,
                     tf.stack(((2 * extent[1] + 1) * (2 * extent[2] + 1), 1)))
        v0 = tf.reshape(v0, tf.stack((n_reps, 1)))

        r = tf.range(-extent[1], extent[1] + 1)
        v1 = tf.expand_dims(r, 1)
        v1 = tf.tile(v1, tf.stack((2 * extent[2] + 1, 2 * extent[0] + 1)))
        v1 = tf.reshape(v1, tf.stack((n_reps, 1)))

        v2 = tf.expand_dims(tf.range(-extent[2], extent[2] + 1), 1)
        v2 = tf.tile(v2,
                     tf.stack((1, (2 * extent[0] + 1) * (2 * extent[1] + 1))))
        v2 = tf.reshape(v2, tf.stack((n_reps, 1)))

        v = tf.cast(tf.concat((v0, v1, v2), axis=1), tf.float32)
        offset = tf.matmul(v, cell)
        offset = tf.expand_dims(offset, 0)

        # add axes
        positions = tf.expand_dims(positions, 1)
        rpos = positions + offset
        rpos = tf.expand_dims(rpos, 0)
        positions = tf.expand_dims(positions, 1)

        euclid_dist = tf.sqrt(
            tf.reduce_sum(tf.square(positions - rpos),
                          reduction_indices=3))
    return euclid_dist

def site_rdf(distances, cutoff, step, width, eps=1e-5,
             use_mean=False, lower_cutoff=None):
    with tf.variable_scope('srdf'):
        if lower_cutoff is None:
            vrange = cutoff
        else:
            vrange = cutoff - lower_cutoff
        distances = tf.expand_dims(distances, -1)
        n_centers = np.ceil(vrange / step)
        gap = vrange - n_centers * step
        n_centers = int(n_centers)

        if lower_cutoff is None:
            centers = tf.linspace(0., cutoff - gap, n_centers)
        else:
            centers = tf.linspace(lower_cutoff + 0.5 * gap, cutoff - 0.5 * gap,
                                  n_centers)
        centers = tf.reshape(centers, (1, 1, 1, -1))

        gamma = -0.5 / width / step ** 2

        rdf = tf.exp(gamma * (distances - centers) ** 2)

        mask = tf.cast(distances >= eps, tf.float32)
        rdf *= mask
        rdf = tf.reduce_sum(rdf, 2)
        if use_mean:
            N = tf.reduce_sum(mask, 2)
            N = tf.maximum(N, 1)
            rdf /= N

        new_shape = [None, None, n_centers]
        rdf.set_shape(new_shape)

    return rdf




class Model(object):
    '''
    the base model for train restore 
    '''
    def __init__(self, model_dir,
                       preprocess_func=None,
                       model_func=None,
                       **config):
        self.reuse = None
        self.preprocess_func = preprocess_func
        self.model_func = model_func

        self.model_dir = model_dir
        self.config = config
        self.config_path = os.path.join(self.model_dir, 'config.npz')

        if not os.path.exists(self.model_dir):
            os.makedir(self.model_dir)

        if os.path.exists(self.config_path):
            logging.warning('Config file exists in model directory\n Config arguments will be overwritten!')
            self.from_config(self.config_path)
        else:
            self.to_config(self.config_path)

        with tf.variable_scope(None, defualt_name=self.__class__.__name__) as scope:
            self.scope = scope
        
        self.saver = None

    def __getattr__(self, key):
        if key in self.config.keys():
            return self.config[key]
        raise AttributeError

    def to_config(self):
        np.savez_compressed(config_path, **self.config)

    def from_config(self, config_path):
        cfg = np.load(config_path)
        for k,v in cfg.items():
            if v.shape == ():
                v = v.item()
            # fix the condition c = np.array(0), c.item()
            self.config[k] = v

    def _preprocessor(self, features):
        '''
        '''
        if self.preprocess_func is None:
            return features
        return self.preprocess_func(features)

    def _model(self, features):
        '''
        Abstract Function:
        return a dict
        '''
        if self.model_func is None:
            raise NotImplementedError
        return self.model_func(features)

    def init_model(self):
        pass

    def store(self, sess, iteration, name='best'):
        checkpoint_path = os.path.join(self.model_dir, name)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if self.saver is None:
            raise ValueError('Saver is not initialized.\n \
                Build the model by calling get_output before storing it.')
        self.saver.save(sess, os.path.join(checkpoint_path, name), iteration)

    def restore(self,sess, name='best', iteration=None):
        '''
        check if we have already stored the check point.
        if we have:
             use self.saver.restore
        else:
            raise Value Error , because we haven't initialized saver.
        return: the latest iteration number 
        '''
        checkpoint_path = os.path.join(self.model_dir, name)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if self.saver is None:
            raise ValueError('''
                Saver is not initialized.
                Build the model by callinmg get_output.
                Before restoring it''')

        if iteration is None:
            check_point = tf.train.latest_checkpoint(checkpoint_path)
        else:
            check_point = os.path.join(checkpoint_path, 'name-{}'.format(iteration))
        logging.info('Restoring {}'.format(check_point))

        self.saver.restore(sess, check_point)
        start_iter = int(check_point.split('-')[-1])
        return start_iter


    def get_output(self, features,
                            is_training,
                            batch_size=None,
                            num_batch_threads=1):

        '''
        features: dict
        '''
        with tf.variable_scope(self.scope, reuse=self.reuse):
            with tf.variable_scope('preprocessing'):
                features = self._preprocessor(features)

            with tf.variable_scope('batching'):
                if batch_size is None:
                    features = {
                    k: tf.expend_dims(v, 0) for k,v in features.items()
                    }
                else:
                    order_names = list(features.keys())
                    order_values = [features[key] for key in order_names]
                    out_shapes = [myshape(v) for v in order_values]
                    # tf.train.batch change a list of numpy array to a list of Tensor
                    tensor_features = tf.train.batch(
                        order_values, batch_size, dynamic_pad=True, # dynamic_pad=True means shape of tensor can change
                        shapes=out_shapes, num_threads=num_batch_threads)
                    # features in above is a list of tensor or a tensor
                    features = dict(zip(order_names, tensor_features))

            with tf.variable_scope('model'):
                self.init_model()
                features['is_training'] = is_training
                output = self._model(features)
                # output is a dict
                features.update(output)
        
        if self.saver is None:
            model_vars = [v for v in tf.global_variables() if v.name.startswith(self.scope.name)]
            logging.info('self.scope.name is {}'.forma(self.scope.name))
            scope_name_len = len(self.scope.name)
            var_names = [v.name[scope_name_len:] for v in model_vars]
            vdict = dict(zip(var_names, model_vars))
            self.saver = tf.train.Saver(vdict)

        self.reuse = True
        return features


class DTNN(Model):
    '''
    Deep Tensor Neural Network (DTNN)

    DTNN receives molecular structures through a vector of atomic `numbers`
    and a matrix of atomic `positions` ensuring rotational and
    translational invariance by construction.
    Each atom is represented by a coefficient vector that
    is repeatedly refined by pairwise interactions with the surrounding atoms.

    For a detailed description, see [1].

    :param str model_dir: path to location of the model
    :param int n_basis: number of basis functions describing an atom
    :param int n_factors: number of factors in tensor low-rank approximation
    :param int n_interactions: number of interaction passes
    :param float mu: mean energy per atom
    :param float std: std. dev. of energies per atom
    :param float cutoff: distance cutoff
    :param float rdf_spacing: gap between Gaussians in distance basis
    :param bool per_atom: `true` if predicted is normalized to the number
                           of atoms
    :param ndarray atom_ref: array of reference energies of single atoms
    :param int max_atomic_number: the highest, occuring atomic number
                                  in the data

    References
    ----------
    .. [1] K.T. Schutt. F. Arbabzadah. S. Chmiela, K.-R. Muller, A. Tkatchenko:
           Quantum-chemical Insights from Deep Tensor Neural Networks.
           Nature Communications 8. 13890 (2017)
           http://dx.doi.org/10.1038/ncomms13890
    
    '''
    def __init__(self, model_dir,
                       n_basis=30,
                       n_factors=60,
                       n_interactions=3,
                       mu=0.0,
                       std=1.0,
                       cutoff=20.0,
                       rdf_spacing=0.2,
                       per_atom=False,
                       atom_ref=None,
                       max_atomic_number=20):
        super(DTNN, self).__init__(
            model_dir,
            n_basis=n_basis,
            n_factors=n_factors,
            n_interactions=n_interactions,
            mu=mu,
            std=std,
            cutoff=cutoff,
            rdf_spacing=rdf_spacing,
            per_atom=per_atom,
            atom_ref=atom_ref,
            max_atomic_number=max_atomic_number)

    def _preprocessor(self, features):
        positions = features['positions']
        # what is positions

        pbc = features['pbc']
        # what is pbc ?

        cell = features['cell']
        # what is cell ?

        distance = inter_atomic_distances(positions, cell, pbc, self.cutoff)

        features['srdf'] = site_rdf(distance, self.cutoff, self.rdf_spacing, 1.0)

        return features

    def _model(self, features):
        c_z = features['numbers']
        c = features['srdf']

        # masking
        mask = tf.cast(tf.expand_dims(c_z, 1) * tf.expand_dims(c_z, 2),
            tf.float32)
        diag = tf.matrix_diag_part(mask)
        diag = tf.ones_like(diag)
        offdiag = 1 - tf.matrix_diag(diag)
        mask *= offdiag
        mask = tf.expand_dims(mask, -1)

        I = np.eye(self.max_z).astype(np.float32)
        zz = tf.nn.embedding_lookup(I, c_z)





                    





