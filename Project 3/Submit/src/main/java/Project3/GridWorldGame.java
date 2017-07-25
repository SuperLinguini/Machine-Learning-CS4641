package Project3;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class GridWorldGame {
    private GridWorldDomain gwdg;
    private OOSADomain domain;
    private TerminalFunction tf;
    private StateConditionTest goalCondition;
    private State initialState;
    private HashableStateFactory hashingFactory;
    private SimulatedEnvironment env;
    private double discount;
    private int size;


    public GridWorldGame(){
        gwdg = new GridWorldDomain(11, 11);
        gwdg.setMapToFourRooms();
        tf = new GridWorldTerminalFunction(10, 10);
        gwdg.setTf(tf);
        goalCondition = new TFGoalCondition(tf);
        domain = gwdg.generateDomain();

        initialState = new GridWorldState(new GridAgent(0, 0), new GridLocation(10, 10, "loc0"));
        hashingFactory = new SimpleHashableStateFactory();

        env = new SimulatedEnvironment(domain, initialState);

        this.discount = .99;
        this.size = 11;


//        VisualActionObserver observer = new VisualActionObserver(domain,
//        	GridWorldVisualizer.getVisualizer(gwdg.getMap()));
//        observer.initGUI();
//        env.addObservers(observer);
    }

    public GridWorldGame(int size, double goalReward, double defaultReward,
                         double discount, double successProb){
        gwdg = new GridWorldDomain(size, size);
        gwdg.setMapToFourRooms();
        gwdg.setProbSucceedTransitionDynamics(successProb);
        tf = new GridWorldTerminalFunction(size - 1, size - 1);
        gwdg.setTf(tf);
        goalCondition = new TFGoalCondition(tf);
        domain = gwdg.generateDomain();

        initialState = new GridWorldState(new GridAgent(0, 0), new GridLocation(size - 1, size - 1, "loc0"));
        hashingFactory = new SimpleHashableStateFactory();

        env = new SimulatedEnvironment(domain, initialState);
        this.discount = discount;
        this.size = size;

        ((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, goalReward, -0.1));

//        VisualActionObserver observer = new VisualActionObserver(domain,
//        	GridWorldVisualizer.getVisualizer(gwdg.getMap()));
//        observer.initGUI();
//        env.addObservers(observer);
    }


    public void visualize(String outputpath){
        Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
        new EpisodeSequenceVisualizer(v, domain, outputpath);
    }

    public void BFSExample(String outputPath){

        DeterministicPlanner planner = new BFS(domain, goalCondition, hashingFactory);
        Policy p = planner.planFromState(initialState);
        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "bfs");

    }

    public void DFSExample(String outputPath){

        DeterministicPlanner planner = new DFS(domain, goalCondition, hashingFactory);
        Policy p = planner.planFromState(initialState);
        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "dfs");

    }

    public void AStarExample(String outputPath){

        Heuristic mdistHeuristic = new Heuristic() {

            public double h(State s) {
                GridAgent a = ((GridWorldState)s).agent;
                double mdist = Math.abs(a.x-10) + Math.abs(a.y-10);

                return -mdist;
            }
        };

        DeterministicPlanner planner = new AStar(domain, goalCondition,
                hashingFactory, mdistHeuristic);
        Policy p = planner.planFromState(initialState);

        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "astar");

    }

    public void valueIterationExample(String outputPath) throws IOException {

        Planner planner = new ValueIteration(domain, discount, hashingFactory, 0.001, 100000);

        long startTime = System.currentTimeMillis();
        System.out.println("Start");
        Policy p = planner.planFromState(initialState);
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("End");
        System.out.println("Time Elapsed: " + estimatedTime + " ms");

        PolicyUtils.rollout(p, initialState, domain.getModel(), 200).write(outputPath + "vi");

        simpleValueFunctionVis((ValueFunction)planner, p, outputPath + "vi");
        //manualValueFunctionVis((ValueFunction)planner, p);

    }


    public void policyIterationExample(String outputPath) throws IOException {

        Planner planner = new PolicyIteration(domain, discount, hashingFactory, .001, 100000, 100000);

        long startTime = System.currentTimeMillis();
        System.out.println("Start");
        Policy p = planner.planFromState(initialState);
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("End");
        System.out.println("Time Elapsed: " + estimatedTime + " ms");

        PolicyUtils.rollout(p, initialState, domain.getModel(), 200).write(outputPath + "pi");

        simpleValueFunctionVis((ValueFunction)planner, p, outputPath + "pi");
    }


    public void qLearningExample(String outputPath){

        LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0., 1.);

        //run learning for 50 episodes
        for(int i = 0; i < 50; i++){
            Episode e = agent.runLearningEpisode(env);

            e.write(outputPath + "ql_" + i);
            System.out.println(i + ": " + e.maxTimeStep());

            //reset environment for next learning episode
            env.resetEnvironment();
        }

    }


    public void sarsaLearningExample(String outputPath){

        LearningAgent agent = new SarsaLam(domain, 0.99, hashingFactory, 0., 0.5, 0.3);

        //run learning for 50 episodes
        for(int i = 0; i < 50; i++){
            Episode e = agent.runLearningEpisode(env);

            e.write(outputPath + "sarsa_" + i);
            System.out.println(i + ": " + e.maxTimeStep());

            //reset environment for next learning episode
            env.resetEnvironment();
        }

    }

    public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p, String outputPath) throws IOException{

        List<State> allStates = StateReachability.getReachableStates(
                initialState, domain, hashingFactory);
        ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
                allStates, this.size, this.size, valueFunction, p);
        gui.initGUI();

        BufferedImage image = new BufferedImage(gui.getWidth(), gui.getHeight(), BufferedImage.TYPE_INT_ARGB);
        Graphics g = image.getGraphics();
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        gui.printAll(g);
        ImageIO.write(image, "png", new File(outputPath + ".png"));
    }

    public void manualValueFunctionVis(ValueFunction valueFunction, Policy p){

        List<State> allStates = StateReachability.getReachableStates(
                initialState, domain, hashingFactory);

        //define color function
        LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
        rb.addNextLandMark(0., Color.RED);
        rb.addNextLandMark(1., Color.BLUE);

        //define a 2D painter of state values,
        //specifying which attributes correspond to the x and y coordinates of the canvas
        StateValuePainter2D svp = new StateValuePainter2D(rb);
        svp.setXYKeys("agent:x", "agent:y",
                new VariableDomain(0, 11), new VariableDomain(0, 11),
                1, 1);

        //create our ValueFunctionVisualizer that paints for all states
        //using the ValueFunction source and the state value painter we defined
        ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(
                allStates, svp, valueFunction);

        //define a policy painter that uses arrow glyphs for each of the grid world actions
        PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
        spp.setXYKeys("agent:x", "agent:y", new VariableDomain(0, 11),
                new VariableDomain(0, 11),
                1, 1);

        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_NORTH, new ArrowActionGlyph(0));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_SOUTH, new ArrowActionGlyph(1));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_EAST, new ArrowActionGlyph(2));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_WEST, new ArrowActionGlyph(3));
        spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);


        //add our policy renderer to it
        gui.setSpp(spp);
        gui.setPolicy(p);

        //set the background color for places where states are not rendered to grey
        gui.setBgColor(Color.GRAY);

        //start it
        gui.initGUI();
    }


    public void experimentAndPlotter(String outputPath, final double qInit, final double learningRate){

        //different reward function for more structured performance plots
        ((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, 5.0, -0.1));

        /**
         * Create factories for Q-learning agent and SARSA agent to compare
         */
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "Q-Learning";
            }


            public LearningAgent generateAgent() {
                return new QLearning(domain, 0.99, hashingFactory, qInit, learningRate);
            }
        };

        LearningAgentFactory sarsaLearningFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "SARSA";
            }


            public LearningAgent generateAgent() {
                return new SarsaLam(domain, 0.99, hashingFactory, qInit, learningRate, 1.);
            }
        };

        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(
                env, 5, 2000, qLearningFactory); // sarsaLearningFactory
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

    public void epsilonComparison(String outputPath, final double qInit, final double learningRate, final double epsilon){

        //different reward function for more interesting results
//		((SimulatedEnvironment)env).setRf(new GoalBasedRF(this.goalCondition, 5.0, -0.1));

        /**
         * Create factories for Q-learning agent with different learning policies to compare
         */
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {
            public String getAgentName() {
                return "Q-Learning";
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


    public static void main(String[] args) {
//        GridWorldGame example = new GridWorldGame();
        GridWorldGame example = new GridWorldGame(11, 5, -0.1, 0.99, 0.99);
        String outputPath = "./output/discount/";

//        example.BFSExample(outputPath);
//        example.DFSExample(outputPath);
//        example.AStarExample(outputPath);
//        example.valueIterationExample(outputPath);
//        example.qLearningExample(outputPath);
//        example.sarsaLearningExample(outputPath);
//
//        example.experimentAndPlotter();

//        example.visualize(outputPath);

        example.visualize(outputPath);
    }

}
