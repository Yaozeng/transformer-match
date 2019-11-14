import tensorflow as tf
from modeling_tf_roberta import TFRobertaForSequenceClassification
from optimization2 import AdamWeightDecayOptimizer
import time
import logging
import json
class BertModel(object):
    def __init__(self,num_train_steps,num_warmup,args):
        self.logger=logging.getLogger("RobertaAlignModel")
        self.learning_rate1=args.learning_rate1
        self.learning_rate2 = args.learning_rate2
        self.log_every_n_step=args.log_every_n_step
        self.num_train_steps = num_train_steps
        self.num_warm_up = num_warmup
        sess_config=tf.ConfigProto()
        sess_config.gpu_options.allow_growth=True
        self.sess=tf.Session(config=sess_config)
        self._build_graph()

        self.saver=tf.train.Saver(max_to_keep=10)
        self.train_writer=tf.summary.FileWriter(args.summary_dir,self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
    def _build_graph(self):
        start_t = time.time()
        self._setup_placeholders()
        self._Main()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
    def _setup_placeholders(self):
        self.input_ids=tf.placeholder(tf.int32,[None,None],"input_ids")
        self.attention_masks=tf.placeholder(tf.int32,[None,None],"attention_masks")
        self.align_masks=tf.placeholder(tf.int32,[None,None,None],"align_masks")
        self.label=tf.placeholder(tf.int32,[None])
        self.global_step=tf.get_variable("global_step", shape=[], dtype=tf.int32,initializer=tf.constant_initializer(0), trainable=False)
    def _Main(self):
        self.model=TFRobertaForSequenceClassification.from_pretrained("./pretrained/robertabase/roberta-base-tf_model.h5")
        self.output=self.model(inputs=(self.input_ids,self.attention_masks,self.align_masks,self.label))
    def _compute_loss(self):
        self.logits=self.output[0]
        self.probabilities = tf.nn.softmax(self.logits, axis=-1)
        log_probs = tf.nn.log_softmax(self.logits, axis=-1)

        one_hot_labels = tf.one_hot(self.label, depth=tf.shape(self.logits)[1], dtype=tf.float32,axis=1)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        self.loss = tf.reduce_mean(per_example_loss)
    def _create_train_op(self):
        # global_step = tf.train.get_or_create_global_step()
        learning_rate1 = tf.constant(value=self.learning_rate1, shape=[], dtype=tf.float32)
        learning_rate2 = tf.constant(value=self.learning_rate2, shape=[], dtype=tf.float32)

        learning_rate1 = tf.train.polynomial_decay(
            learning_rate1,
            self.global_step,
            self.num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)
        learning_rate2 = tf.train.polynomial_decay(
            learning_rate2,
            self.global_step,
            self.num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)

        # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
        # learning rate will be `global_step/num_warmup_steps * init_lr`.
        if self.num_warm_up:
            global_steps_int = tf.cast(self.global_step, tf.int32)
            warmup_steps_int = tf.constant(self.num_warm_up, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate1 = self.learning_rate1 * warmup_percent_done
            warmup_learning_rate2=self.learning_rate2 * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)

            learning_rate1 = (
                    (1.0 - is_warmup) * learning_rate1 + is_warmup * warmup_learning_rate1)
            learning_rate2 = (
                    (1.0 - is_warmup) * learning_rate2 + is_warmup * warmup_learning_rate2)

        self.current_learning_rate1 = learning_rate1
        self.current_learning_rate2=learning_rate2

        self.optimizer1 = AdamWeightDecayOptimizer(learning_rate=learning_rate1, weight_decay_rate=0.1, beta_1=0.9,
                                                      beta_2=0.999, epsilon=1e-6,
                                                      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
        self.optimizer2 = AdamWeightDecayOptimizer(learning_rate=learning_rate2, weight_decay_rate=0.1, beta_1=0.9,
                                                   beta_2=0.999, epsilon=1e-6,
                                                   exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
        tvars=tf.trainable_variables()
        vars1=[v for v in tvars if "classifier" in v.name or "addition_layer" in v.name]
        vars2=[v for v in tvars if "classifier" not in v.name and "addition_layer" not in v.name]
        grads=tf.gradients(self.loss,vars1+vars2)

        grad_var_pairs1 = zip(grads[:len(vars1)], vars1)
        train_op = self.optimizer1.apply_gradients(grad_var_pairs1, name='apply_grad', global_step=self.global_step)

        new_global_step = self.global_step + 1
        train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])
        self.train_op = train_op
    def train(self,train_dataset,batch_size,):
        dataset = train_dataset.repeat().shuffle(1000).batch(batch_size)
        itertor = dataset.make_one_shot_iterator()
        inputs, label = itertor.get_next()
        for step in range(1,self.num_train_steps+1):
            feed_dict={
                self.input_ids:inputs["input_ids"],
                self.attention_masks:inputs["attention_mask"],
                self.align_masks:inputs["align_mask"],
                self.label:label
            }
            total_step=self.sess.run(self.global_step)+1
            _,loss,lr1,lr2=self.sess.run([self.train_op,self.loss,self.learning_rate1,self.learning_rate2],feed_dict)
            tf.summary.scalar('learning_rate1', lr1)
            tf.summary.scalar('learning_rate2', lr2)
            tf.summary.scalar('loss', loss)
            summary=tf.summary.merge_all()
            self.train_writer.add_summary(summary,total_step)





