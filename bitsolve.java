package edu.brown.cs.burlap.tutorials.domain.simple;

//import burlap.behavior.policy.Policy;
//import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
//import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
//import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
//import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
//import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
//import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
//import burlap.behavior.singleagent.planning.Planner;
//import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
//import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
//import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
//import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
//import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
//import burlap.mdp.core.TerminalFunction;
//import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
//import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
//import burlap.mdp.singleagent.model.FactoredModel;
//import burlap.mdp.singleagent.model.RewardFunction;
//import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;
//import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;

import java.io.PrintWriter;
//import java.util.List;


/**
 * @author James MacGlashan.
 */
public class bitsolve {

    public static void main(String [] args) throws Exception {

        DrillModel gw = new DrillModel();
        //PrintWriter writer  = new PrintWriter("exp2.csv");
        PrintWriter writer1  = new PrintWriter("exp6.csv");
        PrintWriter writer2  = new PrintWriter("exp7.csv");
        PrintWriter writer3  = new PrintWriter("exp8.csv");
        PrintWriter writer4  = new PrintWriter("exp9.csv");

        //TerminalFunction tf = new DrillModel.ExampleTF(8000);
        //RewardFunction rf = new DrillModel.ExampleRF(8000);
        //StateConditionTest goalCondition = new TFGoalCondition(tf);
        int[] depth = new int[]{6512,
                6607
                ,6793
                ,7358
                , 7449
                , 7570
                ,7865
                ,8106
                ,8296
                ,8409
                ,8615
                ,8883
                ,8988
                ,9025
                ,9285
                ,9522
                ,9897
                ,9905
                ,10252
                ,10364
                ,10372
                ,10394
                ,10599
                ,10639
                ,10649
                ,10691
                ,10714
                ,10720
                ,10763
                ,10780
                ,10786
                ,10793
                ,10871
                ,10876
                ,10887
                ,11014
                ,11017
                ,11085
                ,11121
                ,11180};
        gw.setGoalLocation(11180);
        //for(int i=0;i<80;i++){
            //depth[i]=100*(i+1);
        //}
        final SADomain domain = gw.generateDomain();
        //ends when the agent reaches a location
        State initialState = new BitState(6319, 6512, 0,0.0,1);
        SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);


        //set up the state hashing system for looking up states
        final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

        //DeterministicPlanner planner = new BFS(domain, goalCondition, hashingFactory);
        //Policy p = planner.planFromState(initialState);
        //Planner planner = new ValueIteration(domain, 0.99, hashingFactory, 0.001, 1);
        //Policy p = planner.planFromState(initialState);
        //List<Action> actionSequence = PolicyUtils.rollout(p, initialState, domain.getModel()).actionSequence;
        //List<State> states = PolicyUtils.rollout(p,initialState,domain.getModel()).stateSequence;

        //System.out.println(actionSequence);
        //System.out.println(states);
        //System.out.println( PolicyUtils.rollout(p,initialState,domain.getModel()).numActions());
        //System.out.println( PolicyUtils.rollout(p,initialState,domain.getModel()).rewardSequence);
        LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 10, 1);

        //StringBuilder sb = new StringBuilder();
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        StringBuilder sb3 = new StringBuilder();
        StringBuilder sb4 = new StringBuilder();

        sb1.append("Counter");
        sb1.append(',');
        sb1.append("Cost");
        sb1.append(',');

        for(int i=0;i<depth.length-1;i++){
            sb1.append(depth[i]).append(" ft");
            sb1.append(',');
        }
        sb1.append('\n');

        sb2.append("Counter");
        sb2.append(',');
        sb2.append("Cost");
        sb2.append(',');

        for(int i=0;i<depth.length-1;i++){
            sb2.append(depth[i]).append(" ft");
            sb2.append(',');
        }
        sb2.append('\n');

        sb3.append("Counter");
        sb3.append(',');
        sb3.append("Cost");
        sb3.append(',');

        for(int i=0;i<depth.length-1;i++){
            sb3.append(depth[i]).append(" ft");
            sb3.append(',');
        }
        sb3.append('\n');

        sb4.append("Counter");
        sb4.append(',');
        sb4.append("Cost");
        sb4.append(',');

        for(int i=0;i<depth.length-1;i++){
            sb4.append(depth[i]).append(" ft");
            sb4.append(',');
        }
        sb4.append('\n');


        //run learning for 50 episodes
        for (int i = 0; i < 200000; i++) {
            System.out.println(i);

            Episode e = agent.runLearningEpisode(env);
            Object[] actseq = e.actionSequence.toArray();
            //System.out.println(e.stateSequence);

            if(i>=150000 && i<160000){

                sb1.append(i);
                sb1.append(',');
                sb1.append("$").append(-1 * e.reward(40));
                sb1.append(',');
                for (int j=0;j<actseq.length;j++) {
                    sb1.append(actseq[j]);
                    sb1.append(',');
                }

                sb1.append('\n');}
            if(i>=160000 && i<170000){
                if(i==150000){
                writer1.write(sb1.toString());
                writer1.close();}
                sb2.append(i);
                sb2.append(',');
                sb2.append("$").append(-1 * e.reward(40));
                sb2.append(',');
                for (Object anActseq : actseq) {
                    sb2.append(anActseq.toString());
                    sb2.append(',');
                }

                sb2.append('\n');}
            if(i>=170000 && i<180000){
                if(i==170000){
                writer2.write(sb2.toString());
                writer2.close();}
                sb3.append(i);
                sb3.append(',');
                sb3.append("$").append(-1 * e.reward(40));
                sb3.append(',');
                for (Object anActseq : actseq) {
                    sb3.append(anActseq.toString());
                    sb3.append(',');
                }

                sb3.append('\n');}
            if(i>=180000){
                if(i==180000){
                writer3.write(sb3.toString());
                writer3.close();}
                sb4.append(i);
                sb4.append(',');
                sb4.append("$").append(-1 * e.reward(40));
                sb4.append(',');
                for (Object anActseq : actseq) {
                    sb4.append(anActseq.toString());
                    sb4.append(',');
                }

                sb4.append('\n');}

            //reset environment for next learning episode
            env.resetEnvironment();

        }



        writer4.write(sb4.toString());
        writer4.close();
        System.out.println("Done lil bih");
        //LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

            //public String getAgentName() {
                //return "Q-Learning";
            //}


            //public LearningAgent generateAgent() {
                //return new QLearning(domain, 0.99, hashingFactory, 10, 1);
            //}
        //};




        //LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env, 1, 00000, qLearningFactory);
        //exp.setUpPlottingConfiguration(500, 250, 2, 1000,
                //TrialMode.MOST_RECENT_AND_AVERAGE,
                //PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
                //PerformanceMetric.AVERAGE_EPISODE_REWARD);

        //exp.startExperiment();

        //exp.writeStepAndEpisodeDataToCSV("expData");

    }

    }





