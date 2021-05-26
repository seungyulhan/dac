from numbers import Number

import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from .base import RLAlgorithm

EPS = 1e-20

class DAC(RLAlgorithm, Serializable):
    """
    Diversity Actor-Critic (DAC)
    """

    def __init__(
            self,
            base_kwargs,
            env,
            eval_env,
            policy,
            initial_exploration_policy,
            qf1,
            qf2,
            vf,
            rf,
            pool,
            af=None,
            plotter=None,
            ctrl_coef=2.0,
            fixed_alpha=0.5,
            lr=3e-3,
            scale_reward=1,
            discount=0.99,
            tau=0.01,
            target_update_interval=1,
            action_prior='uniform',

            save_full_state=False,
    ):
        """
        Args:
            base_kwargs (dict): dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.

            env (`rllab.Env`): rllab environment object.
            policy: (`rllab.NNPolicy`): A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.

            qf1 (`valuefunction`): First Q-function approximator.
            qf2 (`valuefunction`): Second Q-function approximator. Usage of two
                Q-functions improves performance by reducing overestimation
                bias.
            vf (`ValueFunction`): Soft value function approximator.
            rf (`RatioFunction`): Ratio function approximator.
            af (`AlphaFunction`): Alpha function for alpha adaptation.

            pool (`PoolBase`): Replay buffer to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.

            ctrl_coef ('float') : Control coefficient c for alpha adaptation.
            fixed_alpha ('float') : Alpha for fixed alpha setting.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            save_full_state (`bool`): If True, save the full class in the
                snapshot. See `self.get_snapshot` for more information.
        """

        Serializable.quick_init(self, locals())
        super(DAC, self).__init__(**base_kwargs)

        self._env = env
        self._eval_env = eval_env
        self._policy = policy
        self._initial_exploration_policy = initial_exploration_policy
        self._qf1 = qf1
        self._qf2 = qf2
        self._vf = vf
        self._rf = rf
        self._af = af
        if self._af is not None:
            self._alpha_adapt = True
        else:
            self._alpha_adapt = False
        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._qf_lr = lr
        self._vf_lr = lr
        self._rf_lr = lr
        self._af_lr = lr
        self._scale_reward = scale_reward
        self._fixed_alpha=fixed_alpha
        self._c = - ctrl_coef * np.prod(self._env.action_space.shape)
        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        # Reparameterize parameter must match between the algorithm and the
        # policy actions are sampled from.

        self._save_full_state = save_full_state

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim

        self._training_ops = list()

        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_target_ops()

        # Initialize all uninitialized variables. This prevents initializing
        # pre-trained policy and qf and vf variables.
        uninit_vars = []
        for var in tf.global_variables():
            try:
                self._sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        self._sess.run(tf.variables_initializer(uninit_vars))


    @overrides
    def train(self):
        """Initiate training of the DAC instance."""

        self._train(self._env, self._eval_env, self._policy, self._initial_exploration_policy, self._pool)

    def _init_placeholders(self):
        """Create input placeholders for the DAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_pl = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Da),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

    @property
    def scale_reward(self):
        if callable(self._scale_reward):
            return self._scale_reward(self._iteration_pl)
        elif isinstance(self._scale_reward, Number):
            return self._scale_reward

        raise ValueError(
            'scale_reward must be either callable or scalar')

    def _init_critic_update(self):
        """
        Create minimization operation for critic Q-function.
        """

        self._qf1_t = self._qf1.get_output_for(
            self._observations_ph, self._actions_ph, reuse=True)  # N
        self._qf2_t = self._qf2.get_output_for(
            self._observations_ph, self._actions_ph, reuse=True)  # N

        with tf.variable_scope('target'):
            vf_next_target_t = self._vf.get_output_for(self._next_observations_ph)  # N
            self._vf_target_params = self._vf.get_params_internal()

        ys = tf.stop_gradient(
            self.scale_reward * self._rewards_ph +
            (1 - self._terminals_ph) * self._discount * vf_next_target_t
        )  # N

        self._td_loss1_t = 0.5 * tf.reduce_mean((ys - self._qf1_t)**2)
        self._td_loss2_t = 0.5 * tf.reduce_mean((ys - self._qf2_t)**2)

        qf1_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=self._td_loss1_t,
            var_list=self._qf1.get_params_internal()
        )
        qf2_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=self._td_loss2_t,
            var_list=self._qf2.get_params_internal()
        )

        self._training_ops.append(qf1_train_op)
        self._training_ops.append(qf2_train_op)

    def _init_actor_update(self):
        """
        Create minimization operations for policy and state value functions.
        """

        actions, log_pi = self._policy.actions_for(observations=self._observations_ph,
                                                   with_log_pis=True)
        self._log_pi_q = log_pi_q = self._policy.log_pis_for(self._actions_ph)

        self._vf_t = self._vf.get_output_for(self._observations_ph, reuse=True)  # N
        self.rf_pi_t = rf_pi = self._rf.get_output_for(self._observations_ph, actions, reuse=True)# N
        self.rf_q_t = rf_q = self._rf.get_output_for(self._observations_ph, self._actions_ph, reuse=True)
        log_rf_pi = tf.log(rf_pi + EPS)
        log_rf_q = tf.log(rf_q + EPS)  # N
        log_rf_q_inv = tf.log((1.0-rf_q) + EPS)  # N


        self._vf_params = self._vf.get_params_internal()
        self._rf_params = self._rf.get_params_internal()
        print(self._rf_params)

        D_a = actions.shape.as_list()[-1]
        if self._alpha_adapt: # alpha adaptation
            self._alpha_t = self._af.get_output_for(self._observations_ph, reuse=True)  # N
            self._alpha_params = self._af.get_params_internal()

            alpha_reg_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES,
                scope='alpha')
            alpha_reg_loss = tf.reduce_sum(alpha_reg_losses)

            print(self._alpha_params, alpha_reg_losses)
        else: # fixed alpha
            self._alpha_t = self._fixed_alpha

        log_target1 = self._qf1.get_output_for(
            self._observations_ph, actions, reuse=True)  # N
        log_target2 = self._qf2.get_output_for(
            self._observations_ph, actions, reuse=True)  # N
        min_log_target = tf.minimum(log_target1, log_target2)
        policy_kl_loss = tf.reduce_mean(- log_target1 + self._alpha_t * (-log_rf_pi + log_pi))

        policy_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self._policy.name)
        policy_regularization_loss = tf.reduce_sum(
            policy_regularization_losses)

        policy_loss = (policy_kl_loss
                       + policy_regularization_loss)

        # We update the vf towards the min of two Q-functions in order to
        # reduce overestimation bias from function approximation error.
        self._Div = alpha_skewed_D_JS = self._alpha_t * (log_rf_pi - tf.log(self._alpha_t)) + (1-self._alpha_t) * (log_rf_q_inv - tf.log(1.0-self._alpha_t))
        self._vf_loss_t = 0.5 * tf.reduce_mean((
          self._vf_t
          - tf.stop_gradient(min_log_target + self._alpha_t * (log_rf_pi - tf.log(self._alpha_t) - log_pi)  + (1 - self._alpha_t) * tf.clip_by_value(log_rf_q - tf.log(self._alpha_t) - log_pi_q,-D_a,D_a))
        )**2)

        self._rf_loss_t = - tf.reduce_mean(alpha_skewed_D_JS)

        policy_train_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=policy_loss,
            var_list=self._policy.get_params_internal()
        )

        vf_train_op = tf.train.AdamOptimizer(self._vf_lr).minimize(
            loss=self._vf_loss_t,
            var_list=self._vf_params
        )

        rf_train_op = tf.train.AdamOptimizer(self._rf_lr).minimize(
            loss=self._rf_loss_t,
            var_list=self._rf_params
        )

        self._training_ops.append(policy_train_op)
        self._training_ops.append(vf_train_op)
        self._training_ops.append(rf_train_op)

        if self._alpha_adapt:
            self._alpha_loss_t = tf.reduce_mean(self._alpha_t * (log_rf_pi - tf.log(self._alpha_t) - log_pi)  + (1 - self._alpha_t) * tf.clip_by_value(log_rf_q - tf.log(self._alpha_t) - log_pi_q,-D_a,D_a) - self._alpha_t * self._c)
            alpha_train_op = tf.train.AdamOptimizer(self._af_lr).minimize(
                loss=self._alpha_loss_t + alpha_reg_loss,
                var_list=self._alpha_params
            )
            self._training_ops.append(alpha_train_op)


    def _init_target_ops(self):
        """Create tensorflow operations for updating target value function."""

        source_params = self._vf_params
        target_params = self._vf_target_params

        self._target_ops = [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target_params, source_params)
        ]

    @overrides
    def _init_training(self, env, eval_env, policy, pool):
        super(DAC, self)._init_training(env, eval_env, policy, pool)
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        self._sess.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if iteration is not None:
            feed_dict[self._iteration_pl] = iteration

        return feed_dict

    @overrides
    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)

        if self._alpha_adapt:
            qf1, qf2, vf, div, alpha, rf_pi, rf_q, rf_loss, td_loss1, td_loss2 = self._sess.run(
                (self._qf1_t, self._qf2_t, self._vf_t, self._Div, self._alpha_t, self.rf_pi_t, self.rf_q_t, self._rf_loss_t, self._td_loss1_t, self._td_loss2_t), feed_dict)

            logger.record_tabular('alpha-avg', np.mean(alpha))
            logger.record_tabular('alpha-std', np.std(alpha))
        else:
            qf1, qf2, vf, div, rf_pi, rf_q, rf_loss, td_loss1, td_loss2 = self._sess.run(
                (self._qf1_t, self._qf2_t, self._vf_t, self._Div, self.rf_pi_t, self.rf_q_t, self._rf_loss_t, self._td_loss1_t, self._td_loss2_t), feed_dict)
            logger.record_tabular('alpha', self._alpha_t)


        logger.record_tabular('div-avg', np.mean(div))
        logger.record_tabular('qf1-avg', np.mean(qf1))
        logger.record_tabular('qf1-std', np.std(qf1))
        logger.record_tabular('qf2-avg', np.mean(qf1))
        logger.record_tabular('qf2-std', np.std(qf1))
        logger.record_tabular('mean-qf-diff', np.mean(np.abs(qf1-qf2)))
        logger.record_tabular('vf-avg', np.mean(vf))
        logger.record_tabular('vf-std', np.std(vf))
        logger.record_tabular('rf_pi-avg', np.mean(rf_pi))
        logger.record_tabular('rf_pi-std', np.std(rf_pi))
        logger.record_tabular('rf_q-avg', np.mean(rf_q))
        logger.record_tabular('rf_q-std', np.std(rf_q))
        logger.record_tabular('rf_loss', rf_loss)
        logger.record_tabular('mean-sq-bellman-error1', td_loss1)
        logger.record_tabular('mean-sq-bellman-error2', td_loss2)

        self._policy.log_diagnostics(iteration, batch)
        if self._plotter:
            self._plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the DAC algorithm.

        If `self._save_full_state == True`, returns snapshot of the complete
        DAC instance. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, state value function, and environment instances.
        """

        if self._save_full_state:
            snapshot = {
                'epoch': epoch,
                'algo': self
            }
        else:
            snapshot = {
                'epoch': epoch,
                'policy': self._policy,
                'qf1': self._qf1,
                'qf2': self._qf2,
                'vf': self._vf,
                'rf': self._rf,
                'env': self._env,
            }

        return snapshot

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        d = Serializable.__getstate__(self)
        d.update({
            'qf1-params': self._qf1.get_param_values(),
            'qf2-params': self._qf2.get_param_values(),
            'vf-params': self._vf.get_param_values(),
            'rf-params': self._rf.get_param_values(),
            'policy-params': self._policy.get_param_values(),
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
        })
        return d

    def __setstate__(self, d):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, d)
        self._qf1.set_param_values(d['qf1-params'])
        self._qf2.set_param_values(d['qf2-params'])
        self._vf.set_param_values(d['vf-params'])
        self._rf.set_param_values(d['rf-params'])
        self._policy.set_param_values(d['policy-params'])
        self._pool.__setstate__(d['pool'])
        self._env.__setstate__(d['env'])
