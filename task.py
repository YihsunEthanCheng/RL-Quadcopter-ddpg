import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, 
                 init_pose=None,
                 init_velocities=None, 
                 init_angle_velocities=None,
                 runtime=5., 
                 sel_state_variables = ['pose','v','angular_v'],
                 target_state_variables = None,
                 repeat = 2):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions
                and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z)
                dimensions init_angle_velocities: initial radians/second for 
                each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, 
                              init_angle_velocities, runtime, sel_state_variables) 
        self.action_repeat = repeat
        
 
        # Goal, current state is target state if not specified
        if target_state_variables == None:
            self.target_pose = self.sim.pose
            self.target_velocity = self.sim.v
            self.target_angular_velocity = self.sim.angular_v
            target_state_variables = self.sim.get_state_variables()
        self.target_state_variables = target_state_variables
       
        self.state_size = self.action_repeat * len(target_state_variables)
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

    def get_state_variables(self):
        return self.sim.get_state_variables()
    
    def get_state_labels(self):
        return self.sim.get_state_labels()

    def get_time(self):
        return self.sim.get_time()

    def get_reward(self):
        # quadratic reward function
        dif = self.sim.get_state_variables() - self.target_state_variables
        reward = 5 - dif.dot(dif)/25.0
       
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) 
            # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.get_state_variables())
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.array(self.get_state_variables().tolist() * self.action_repeat) 
        return state