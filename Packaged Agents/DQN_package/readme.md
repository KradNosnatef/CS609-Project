### For training and testing model
use $ python3 Train_and_test_DQN_model.py
run the command above and see usage of the py, then follow the usage to train your own model
recommend to use my completely trained model
### For using packaged agent
the packaged agent is set to use policy_net_path='policy_net.pt' as default, but you can use other path by setting policy_net_path when initing the object
the only safe-for-public-call function of a DQN_Agent object is agent.act(state), I'm not sure what the consequences of calling other functions will be.