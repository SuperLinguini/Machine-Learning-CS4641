package Project3;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

public class BD {
    private BlockDude bd;
    private OOSADomain domain;
    private State initialState;
    private SimpleHashableStateFactory hashingFactory;
    private TerminalFunction tf;
    private RewardFunction rf;
    private double discount;
    private int level;
    private SimulatedEnvironment env;
    List<Episode> episodes = new ArrayList<Episode>(1000);


    public BD (double goalReward, double defaultReward, double discount, int level){
        this.bd = new BlockDude();
        this.level = level;
        this.domain = bd.generateDomain();
        if(this.level == 1){
            this.initialState = BlockDudeLevelConstructor.getLevel1(domain);
        } else if(this.level == 2) {
            this.initialState = BlockDudeLevelConstructor.getLevel2(domain);
        } else if(this.level == 3) {
            this.initialState = BlockDudeLevelConstructor.getLevel3(domain);
        } else {
            this.initialState = BlockDudeLevelConstructor.getLevel3(domain);
        }
        this.hashingFactory = new SimpleHashableStateFactory();
        tf = new BlockDudeTF();
        rf = new GoalBasedRF(new TFGoalCondition(tf), goalReward, defaultReward);
        bd.setRf(rf);
        bd.setTf(tf);
        this.discount = discount;
        env = new SimulatedEnvironment(domain, initialState);
    }


    public BlockDude getBlockDude() {
        return this.bd;
    }

    public SADomain getDomain() {
        return this.domain;
    }

    public State getInitialState() {
        return this.initialState;
    }

    public SimpleHashableStateFactory getHashingFactory() {
        return this.hashingFactory;
    }

    public TerminalFunction getTerminalFunction() {
        return this.tf;
    }

    public RewardFunction getRewardFunction() {
        return this.rf;
    }

    public void valueIteration(String outputPath) throws IOException{

        Planner planner = new ValueIteration(domain, discount, hashingFactory, 0.001, 1000);
        long startTime = System.currentTimeMillis();
        System.out.println("Start");
        Policy p = planner.planFromState(initialState);
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("End");
        System.out.println("Time Elapsed: " + estimatedTime + " ms");

        PolicyUtils.rollout(p, initialState, domain.getModel(), 200).write(outputPath + "vi");

//        simpleValueFunctionVis((ValueFunction)planner, p);
//        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "_vi");
        //simpleValueFunctionVis((ValueFunction)planner, p, outputPath + "_vi");
    }


    public void policyIteration(String outputPath) throws IOException{

        Planner planner = new PolicyIteration(domain, discount, hashingFactory, 0.001, 100, 100);

        long startTime = System.currentTimeMillis();
        System.out.println("Start");
        Policy p = planner.planFromState(initialState);
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("End");
        System.out.println("Time Elapsed: " + estimatedTime + " ms");

        PolicyUtils.rollout(p, initialState, domain.getModel(), 200).write(outputPath + "pi");
//        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "_pi");

        //simpleValueFunctionVis((ValueFunction)planner, p, outputPath + "_pi");
    }

//    public void QLearning(String outputPath, final double qInit, final double learningRate, final double epsilon){
//        //initial state generator
//        final ConstantStateGenerator sg = new ConstantStateGenerator(initialState);
//        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {
//
//            @Override
//            public String getAgentName() {
//                return "Q-learning";
//            }
//
//            @Override
//            public LearningAgent generateAgent() {
//                return new MyQLearning(domain, discount, hashingFactory, qInit, learningRate,epsilon);
//            }
//        };
//        //define learning environment
//        SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, sg);
//        //define experiment
//        MyLearningAlgorithmExperimenter exp = new MyLearningAlgorithmExperimenter(env,
//                5, 500, qLearningFactory);
//        exp.setUpPlottingConfiguration(500, 500, 2, 1000, TrialMode.MOSTRECENTANDAVERAGE,
//                PerformanceMetric.STEPSPEREPISODE,
//                PerformanceMetric.AVERAGEEPISODEREWARD);
//        //exp.toggleTrialLengthInterpretation(false);
//        //start experiment
//        long startTime = System.currentTimeMillis();
//        System.out.println("Start");
//        exp.startExperiment();
//        long estimatedTime = System.currentTimeMillis() - startTime;
//        System.out.println("End");
//        System.out.println("Time Elapsed: " + estimatedTime + " ms");
//    }

    public void QLearningExample(String outputPath){

        LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0., .99, 1);

        //run learning for 50 episodes
        for(int i = 0; i < 1; i++){
            Episode e = agent.runLearningEpisode(env);

            e.write(outputPath + "ql_" + i);
            System.out.println(i + ": " + e.maxTimeStep());

            //reset environment for next learning episode
            env.resetEnvironment();
        }

    }

    public void experimentAndPlotter(final double qInit){

        //different reward function for more structured performance plots
//        ((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, 5.0, -0.1));

        /**
         * Create factories for Q-learning agent and SARSA agent to compare
         */
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "Q-Learning";
            }


            public LearningAgent generateAgent() {
                return new QLearning(domain, 0.99, hashingFactory, qInit, 0.1, 100);
            }
        };

        LearningAgentFactory sarsaLearningFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "SARSA";
            }


            public LearningAgent generateAgent() {
                return new SarsaLam(domain, 0.99, hashingFactory, qInit, 0.1, 100,1.);
            }
        };

//        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(
//                env, 10, 100, qLearningFactory, sarsaLearningFactory);

        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
                5, 1000, qLearningFactory);//, sarsaLearningFactory);

//        exp.toggleTrialLengthInterpretation(false);

        exp.setUpPlottingConfiguration(500, 250, 2, 1000,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
                PerformanceMetric.AVERAGE_EPISODE_REWARD);


        long startTime = System.currentTimeMillis();
        System.out.println("Start");
        exp.startExperiment();
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("End");
        System.out.println("Time Elapsed: " + estimatedTime + " ms");
        exp.writeStepAndEpisodeDataToCSV("expData");

    }

    public void epsilonComparison(String outputPath, final double qInit, final double learningRate, final double epsilon){
        /**
         * Create factories for Q-learning agent with different learning policies to compare
         */
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {
            public String getAgentName() {
                return "Q-Learning with Epsilon";
            }

            public LearningAgent generateAgent() {
                QLearning q = new QLearning(domain, 0.99, hashingFactory, qInit, learningRate);
                q.setLearningPolicy(new EpsilonGreedy(q, epsilon));;
                return q;
            }
        };

        //set up learning agent experimenter
        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env, 5, 2000,
                qLearningFactory);
        exp.setUpPlottingConfiguration(500, 250, 2, 1000,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.STEPS_PER_EPISODE,
                PerformanceMetric.AVERAGE_EPISODE_REWARD);

        long startTime = System.currentTimeMillis();
        System.out.println("Start");
        exp.startExperiment();
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("End");
        System.out.println("Time Elapsed: " + estimatedTime + " ms");

        exp.writeStepAndEpisodeDataToCSV(outputPath);
    }

    public void visualize() {
        Visualizer v = BlockDudeVisualizer.getVisualizer(bd.getMaxx(), bd.getMaxy());
        new EpisodeSequenceVisualizer(v, domain, episodes);
    }

    public void visualize(String outputPath) {
        Visualizer v = BlockDudeVisualizer.getVisualizer(bd.getMaxx(), bd.getMaxy());
        new EpisodeSequenceVisualizer(v, domain, outputPath);
    }

    public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p) throws IOException{

/*		List<State> allStates = StateReachability.getReachableStates(initialState, 
									(SADomain)domain, hashingFactory);
		MyValueFunctionVisualizerGUI gui = MyGridWorldDomain.getGridWorldValueFunctionVisualization(
											allStates, valueFunction, p);
		gui.initGUI();
		BufferedImage image = new BufferedImage(gui.getWidth(), gui.getHeight(), BufferedImage.TYPE_INT_ARGB);
		Graphics g = image.getGraphics();
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		gui.printAll(g);
		ImageIO.write(image, "png", new File(outputPath + ".png"));*/

//        List<State> allStates = StateReachability.getReachableStates(
//                initialState, domain, hashingFactory);
//        ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
//                allStates, bd.getMaxx(), bd.getMaxy(), valueFunction, p);
//        gui.initGUI();

    }


    public void initGUI(){
        Visualizer v = BlockDudeVisualizer.getVisualizer(bd.getMaxx(), bd.getMaxy());
        VisualExplorer exp = new VisualExplorer(domain, v, initialState);

        exp.addKeyAction("w", BlockDude.ACTION_UP, "");
        exp.addKeyAction("d", BlockDude.ACTION_EAST, "");
        exp.addKeyAction("a", BlockDude.ACTION_WEST, "");
        exp.addKeyAction("s", BlockDude.ACTION_PICKUP, "");
        exp.addKeyAction("x", BlockDude.ACTION_PUT_DOWN, "");

        exp.initGUI();
    }

    /*public static void main(String[] args) throws IOException {
        String outputPath = "./output/qInit/";
        BD bd = new BD(5,-0.1,0.99,3);
//        bd.valueIteration(outputPath);
//        bd.policyIteration(outputPath);
//        bd.experimentAndPlotter();
//        bd.QLearningExample(outputPath);
        bd.visualize(outputPath);
//        bd.initGUI();
    }*/
}