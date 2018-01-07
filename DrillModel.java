package edu.brown.cs.burlap.tutorials.domain.simple;

import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.core.StateTransitionProb;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.UniversalActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.statemodel.FullStateModel;
import burlap.shell.visual.VisualExplorer;
import burlap.visualizer.Visualizer;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerMinMaxScalerSerializer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;



/**
 * @author James MacGlashan.
 */
public class DrillModel implements DomainGenerator {

    public static final String DEPTH_IN = "y1";
    public static final String DEPTH_OUT = "y2";
    public static final String BIT_FOOTAGE = "y";
    public static final String CUMM_TIME = "TIME";
    public static final String BIT_NUMBER = "n";
    public static final String ACTION_CHANGE = "changebit";
    public static final String ACTION_CONTINUE = "continue";



    int targetDepth ;


    protected double [][] map = new double [][] {
            {6319,6512,
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
                    ,11180
            },
            {5,10
                    ,10
                    ,0
                    ,15
                    ,20
                    ,20
                    ,20
                    ,20
                    ,20
                    ,25
                    ,25
                    ,10
                    ,10
                    ,25
                    ,25
                    ,25
                   , 0
                    ,5
                   , 5
                   , 5
                    ,10
                    ,10
                    ,10
                    ,10
                    ,20
                   , 20
                    ,0
                    ,0
                    ,0
                    ,0
                    ,0
                   , 0
                   , 0
                    ,0
                    ,0
                    ,0
                    ,0
                    ,0
                   , 0
                    ,0

            },
            {10,15
                    ,15
                    ,20
                    ,15
                    ,20
                    ,22
                    ,22
                    ,25
                    ,25
                   , 30
                   , 30
                   , 15
                   , 25
                    ,30
                    ,30
                    ,30
                    ,2
                    ,10
                   , 10
                    ,10
                    ,15
                    ,15
                    ,15
                   , 15
                   , 30
                    ,30
                    ,30
                    ,60
                    ,60
                    ,35
                    ,30
                    ,50
                    ,30
                    ,60
                    ,30
                    ,35
                    ,40
                    ,40
                    ,40
                    ,40

            },
            {9.4,9.4
                   , 9.4
                 ,   9.4
                  ,  9.4
                   , 9.4
                   , 9.4
                  ,  9.4
,                    9.5
 ,                   9.6
  ,                  9.6
   ,                 9.7
    ,                9.7
     ,               9.7
      ,              9.8
       ,             9.8
        ,            9.8
         ,           9.8
          ,          9.8
           ,         9.9
            ,        9.9
             ,       9.9
              ,      9.9
               ,     9.9
                ,    9.9
                 ,   9.9
                  ,  10.2
                   , 10
                    ,10
     ,               10
      ,              10
       ,             10.1
        ,            10
         ,           10
          ,          10
           ,         10
            ,        10.1
             ,       10.1
              ,      10.1
               ,     10.1
                ,    10.1
            },
            {70,70
,                    70

  ,                  70
   ,                 70
    ,                70
     ,               70
      ,              70
       ,             70
        ,            70
         ,           70
          ,          70
           ,         70
            ,        75
             ,       75
              ,      75
               ,     67
                ,    68
                 ,   68
                  ,  70
                   , 70
                    ,65
,                    65
 ,                   60
  ,                  60
   ,                 70
    ,                75
     ,               73
      ,              73
       ,             73
        ,            73
         ,           73
          ,          75
           ,         75
            ,        75
             ,       75
              ,      72
               ,     72
                ,    72
                 ,   72
                  ,  72
            },
            {0,4
,                    4

  ,                  4
   ,                 4
    ,                4
     ,               4.5
      ,              4.5
       ,             4
        ,            4
         ,           4
          ,          4
           ,         0
            ,        0
             ,       7
              ,      7
               ,     8
                ,    9
                 ,   9
                  ,  9
                   , 9
                    ,9
,                    9
 ,                   9
  ,                  0
   ,                 7.2
    ,                7.2
     ,               9
      ,              9
       ,             9
        ,            9
         ,           10
          ,          4
           ,         4
            ,        11
             ,       11
              ,      11
               ,     10
                ,    10
                 ,   10
                  ,  10
            },
            {5,4,
                    4
 ,                   4
  ,                  4
   ,                 4
    ,                4.5
     ,               4.5
      ,              4
       ,             6
        ,            6
         ,           6
          ,          0
           ,         0
            ,        8
             ,       8
              ,      10
               ,     9.5
                ,    9.5
                 ,   9
                  ,  9
                   , 9
                    ,9
,                    9
 ,                   0
  ,                  7.2
   ,                 7.2
    ,                9
     ,               9
      ,              9
       ,             9
        ,            10
         ,           4
          ,          4
           ,         11
            ,        11
             ,       11
              ,      10
               ,     12
                ,    12
                 ,   10
            },
            {0,0
,                    0

  ,                  0
   ,                 0
    ,                0
     ,               0
      ,              1
       ,             1
        ,            1
         ,           1
          ,          1
           ,         1
            ,        1
             ,       1
              ,      1
               ,     0
                ,    0
                 ,   0
                  ,  0
                   , 0
                    ,0
,                    1
 ,                   1
  ,                  1
   ,                 1
    ,                1
     ,               1
      ,              1
       ,             1
        ,            1
         ,           1
          ,          1
           ,         1
            ,        1
             ,       1
              ,      1
               ,     1
                ,    0
                 ,   0
                  ,  0
            },
            {250,250
,                    250
  ,                  260
   ,                 260
    ,                260
     ,               270
      ,              270
       ,             285
        ,            285
         ,           285
          ,          290
           ,         300
            ,        300
             ,       310
              ,      310
               ,     310
                ,    320
                 ,   320
                  ,  320
                   , 320
                    ,320
,                    320
 ,                   320
  ,                  320
   ,                 320
    ,                320
     ,               325
      ,              325
       ,             325
        ,            325
         ,           325
          ,          325
           ,         325
            ,        325
             ,       325
              ,      325
               ,     325
                ,    325
                 ,   325
                  ,  325
            },
            {250,245
,                    245

  ,                  255
   ,                 255
    ,                255
     ,               265
      ,              265
       ,             275
        ,            275
         ,           275
          ,          280
           ,         280
            ,        280
             ,       300
              ,      300
               ,     280
                ,    290
                 ,   290
                  ,  290
                   , 290
                    ,290
,                    290
 ,                   290
  ,                  290
   ,                 280
    ,                280
     ,               280
      ,              280
       ,             280
        ,            280
         ,           280
          ,          280
           ,         280
            ,        280
             ,       280
              ,      280
               ,     280
                ,    280
                 ,   280
                  ,  280
            },
            {250,250
,                    250,                   260
  ,                  260
   ,                 260
    ,                260
     ,               270
      ,              270
       ,             280
        ,            280
         ,           280
          ,          285
           ,         285
            ,        285
             ,       295
              ,      295
               ,     290
                ,    300
                 ,   300
                  ,  300
                   , 300
                    ,300
,                    300
 ,                   300
  ,                  300
   ,                 300
    ,                300
     ,               300
      ,              300
       ,             300
        ,            300
         ,           300
          ,          300
           ,         300
            ,        300
             ,       300
              ,      300
               ,     300
                ,    300
                 ,   300
                  ,  300
            },
            {40,40
,                    40

  ,                  40
   ,                 40
    ,                40
     ,               40
      ,              40
       ,             40
        ,            40
         ,           40
          ,          40
           ,         0
            ,        0
             ,       40
              ,      40
               ,     40
                ,    40
                 ,   40
                  ,  40
                   , 40
                    ,40
,                    40
 ,                   40
  ,                  0
   ,                 40
    ,                40
     ,               60
      ,              50
       ,             40
        ,            40
         ,           60
          ,          40
           ,         0
            ,        40
             ,       40
              ,      40
               ,     40
                ,    52.5
                 ,   52.5
                  ,  40
            },
            {0,0
,                    95
 ,                   281
  ,                  846
   ,                 842
    ,                0
     ,               295
      ,              536
       ,             0
        ,            113
         ,           319
          ,          527
           ,         0
            ,        37
             ,       297
              ,      534
               ,     0
                ,    8
                 ,   355
                  ,  467
                   , 475
                    ,497
,                    702
 ,                   742
  ,                  0
   ,                 42
    ,                65
     ,               71
      ,              114
       ,             131
        ,            137
         ,           0
          ,          78
           ,         83
            ,        94
             ,       221
              ,      224
               ,     292
                ,    328
                 ,   387
            }
    };


    public void setGoalLocation(int targetDepth){
        this.targetDepth = targetDepth;

    }


    @Override
    public SADomain generateDomain() {

        SADomain domain = new SADomain();


        domain.addActionTypes(
                new UniversalActionType(ACTION_CHANGE),
                new UniversalActionType(ACTION_CONTINUE));


        DrillStateModel smodel = new DrillStateModel();
        RewardFunction rf = new ExampleRF(this.targetDepth);
        TerminalFunction tf = new ExampleTF(this.targetDepth);

        domain.setModel(new FactoredModel(smodel, rf, tf));

        return domain;
    }


    protected class DrillStateModel implements FullStateModel {


        @Override
        public List<StateTransitionProb> stateTransitions(State s, Action a) {
            BitState gs = (BitState) s;
            BitState bs = (BitState) s;
            int Depth_in = gs.y1;
            double time = gs.TIME;
            int footage = gs.y;
            int curDepth = gs.y2;
            int footagee = bs.y;
            int Depth_inn = bs.y1;
            int act = actionDir(a);
            int bitnum = bs.n;
            int[][] T = new int[2][2];
            T[0][0]=T[1][1]=1;
            T[0][1]=T[1][0]=0;


            List<StateTransitionProb> tps = new ArrayList<>(2);
            StateTransitionProb noChange = null;

            for (int i = 0; i < 2; i++) {

                double[] newPos = this.moveResult(Depth_in, curDepth, footage, Depth_inn, footagee, i,bitnum);
                if (newPos[1] != curDepth) {
                    //new possible outcome
                    BitState ns = gs.copy();
                    ns.y1 = (int)newPos[0];
                    ns.y2 = (int)newPos[1];
                    ns.y = (int)newPos[2];
                    ns.TIME = newPos[3]+time;
                    ns.n = (int)newPos[4];

                        tps.add(new StateTransitionProb(ns, T[i][act]));


                }
                else{

                    if(noChange != null){
                        noChange.p += T[i][act];
                    }
                    else{
                        //otherwise create this new state and transition
                        noChange = new StateTransitionProb(s.copy(), T[i][act]);
                        tps.add(noChange);
                    }
                }


            }


            return tps;
        }

        @Override
        public State sample(State s, Action a) {

            s = s.copy();
            BitState gs = (BitState) s; 
            BitState bs = (BitState) s;
            int Depth_in = gs.y1;
            double time = gs.TIME;
            int footage = gs.y;
            int curDepth = gs.y2;
            int action = actionDir(a);
            int footagee = bs.y;
            int bitnum = bs.n;
            int Depth_inn = bs.y1;
            double[] newPos = this.moveResult(Depth_in, curDepth, footage, Depth_inn, footagee, action,bitnum);

            //set the new position
            gs.y1 = (int)newPos[0];
            gs.y2 = (int)newPos[1];
            gs.y = (int)newPos[2];
            gs.TIME = newPos[3]+time;
            gs.n = (int)newPos[4];


            //return the state we just modified
            return gs;
        }

        protected int actionDir(Action a) {
            int adir = -1;
            if (a.actionName().equals(ACTION_CONTINUE)) {
                adir = 0;

            } else if (a.actionName().equals(ACTION_CHANGE)) {
                adir = 1;
            }

            return adir;
        }


        protected double[] moveResult(int y1, int y2, int y, int y3, int y4, int action,int n) {
            double ROP;
            double[] p = new double[13];
            double[]param ;

            for(int i = 0;i<map[0].length;i++) {
                if (y3 == map[0][i]) {
                    for (int j = 0; j < 13; j++) {
                        p[j] = map[j][i];}
                        y2=(int)map[0][i+1];

                    break;
                }
            }
            param = new double[]{p[1], p[2], p[11], p[3], p[4], y1, y2, 12.25, y, p[5], p[6], p[7], p[8], p[9],p[10]};
                    double PR = estimate(param);
                    if(PR<=0){
                        PR=0.7;
                    }

            if (action == 0) {
                ROP = ((y2 - y3) / PR);
                for (int i = 0; i < map[0].length; i++) {
                    if (y2 == map[0][i]) {
                        y = y + y2 - y1;
                        y1 = y2;


                        break;

                    }


                }
            } else {
                ROP = .2 + ((y2 - y3) / PR);
                n++;
                y = 0;
                y1 = y2;


            }






            return new double[]{y1, y2, y, ROP,n};

        }

        protected int estimate(double[]param) {
            int ROP = 0;
            try {
                File location = new File("MyReg2.zip");
                File norms = new File("normalizers.zip");
                NormalizerMinMaxScalerSerializer serializer = new NormalizerMinMaxScalerSerializer();
                NormalizerMinMaxScaler normalizer = serializer.restore(norms);
                MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(location);


                final INDArray input = Nd4j.create(param);
                final INDArray output = Nd4j.create(new double[]{0});
                normalizer.transform(input);

                INDArray result = net.output(input);
                normalizer.revertLabels(result);
                ROP = result.getInt(0);
                return ROP;
            } catch (IOException e) {
                e.printStackTrace();
            }



            return ROP;
        }
    }





    public static class ExampleRF implements RewardFunction {
        int targetDepth;
        public ExampleRF(int targetDepth){
            this.targetDepth=targetDepth;
        }

        @Override
        public double reward(State s, Action a, State sprime) {
            BitState gs = (BitState) s;


            if(gs.y2==11121 ) {
                return -(gs.n *50000 + 7000*gs.TIME);
            }
            else return 0;



        }


    }

    public static class ExampleTF implements TerminalFunction {

        int targetDepth;

        public ExampleTF(int targetDepth){
            this.targetDepth = targetDepth;

        }

        @Override
        public boolean isTerminal(State s) {

            //get location of agent in next state
            int y2 = (Integer)s.get(DEPTH_OUT);


            //are they at goal location?
            if(y2 == this.targetDepth ){
                return true;
            }

            return false;
        }



    }



}