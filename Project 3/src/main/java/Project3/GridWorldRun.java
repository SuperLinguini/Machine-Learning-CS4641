package Project3;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

public class GridWorldRun {

    public static void main(String[] args) throws IOException {
        int[] size = {5,10,20,40,80};
        double[] prob = {0.99,0.9,0.8,0.6,0.4,0.2};
        double[] reward = {-0.1,1,3,5,10,100};
        double[] discount = {0.99,0.95,0.9,0.8,0.6};
        double[] qInit = {0.3,0.5,1,5,30};

        OutputStream outs = System.out;
        PrintStream dos = new PrintStream(outs);

        String head = "size";
        System.setOut(dos);
        System.out.println(head);
        new File("./output/"+head).mkdirs();
        File file = new File("./output/"+head+"/out_GW"); //Your file
        FileOutputStream fos = new FileOutputStream(file);
        PrintStream ps = new PrintStream(fos);
        System.setOut(ps);
        for (int i = 0; i < size.length; i++) {
            System.out.println("----------------------"+head+"_"+size[i]+"-----------------------");
            GridWorldGame gw = new GridWorldGame(size[i], 5, -0.1, 0.99, 0.99);
            System.out.println("-------------Value Iteration-------------");
            gw.valueIterationExample("./output/"+head+"/GridWorld_"+size[i]);
            System.out.println("-------------Policy Iteration-------------");
            gw.policyIterationExample("./output/"+head+"/GridWorld_"+size[i]);
        }

        head = "prob";
        System.setOut(dos);
        System.out.println(head);
        new File("./output/"+head).mkdirs();
        file = new File("./output/"+head+"/out_GW"); //Your file
        fos = new FileOutputStream(file);
        ps = new PrintStream(fos);
        System.setOut(ps);
        for(int i=0;i<prob.length;i++) {
            System.out.println("----------------------" + head + "_" + prob[i] + "-----------------------");
            GridWorldGame gw = new GridWorldGame(11, 5, -0.1, 0.99, prob[i]);
            System.out.println("-------------Value Iteration-------------");
            gw.valueIterationExample("./output/" + head + "/GridWorld_" + prob[i]);
            System.out.println("-------------Policy Iteration-------------");
            gw.policyIterationExample("./output/" + head + "/GridWorld_" + prob[i]);
        }

        head = "reward";
        System.setOut(dos);
        System.out.println(head);
        new File("./output/"+head).mkdirs();
        file = new File("./output/"+head+"/out_GW"); //Your file
        fos = new FileOutputStream(file);
        ps = new PrintStream(fos);
        System.setOut(ps);
        for (int i = 0; i < reward.length; i++){
            System.out.println("----------------------"+head+"_"+reward[i]+"-----------------------");
            GridWorldGame gw = new GridWorldGame(11, reward[i], -0.1, 0.99, 0.99);
            System.out.println("-------------Value Iteration-------------");
            gw.valueIterationExample("./output/"+head+"/GridWorld_"+reward[i]);
            System.out.println("-------------Policy Iteration-------------");
            gw.policyIterationExample("./output/"+head+"/GridWorld_"+reward[i]);
        }

        head = "discount";
        System.setOut(dos);
        System.out.println(head);
        new File("./output/"+head).mkdirs();
        file = new File("./output/"+head+"/out_GW"); //Your file
        fos = new FileOutputStream(file);
        ps = new PrintStream(fos);
        System.setOut(ps);
        for (int i = 0; i < discount.length; i++){
            System.out.println("----------------------"+head+"_"+discount[i]+"-----------------------");
            GridWorldGame gw = new GridWorldGame(11, 5, -0.1, discount[i], 0.99);
            System.out.println("-------------Value Iteration-------------");
            gw.valueIterationExample("./output/"+head+"/GridWorld_"+discount[i]);
            System.out.println("-------------Policy Iteration-------------");
            gw.policyIterationExample("./output/"+head+"/GridWorld_"+discount[i]);
        }

        head = "qInit";
        System.setOut(dos);
        System.out.println(head);
        new File("./output/"+head).mkdirs();
        file = new File("./output/"+head+"/out_GW"); //Your file
        fos = new FileOutputStream(file);
        ps = new PrintStream(fos);
        System.setOut(ps);
        for(int i=0;i<qInit.length;i++){
            System.out.println("----------------------"+head+"_"+qInit[i]+"-----------------------");
            GridWorldGame gw = new GridWorldGame(11, 5, -0.1, 0.99, 0.99);
            System.out.println("-------------Q Learning-------------");
            gw.experimentAndPlotter("QLearning_qInit_" + i, qInit[i], 0.1);
        }

        head = "discount";
        System.setOut(dos);
        System.out.println(head);
        new File("./output/"+head).mkdirs();
        file = new File("./output/"+head+"/out_GW_QLearning"); //Your file
        fos = new FileOutputStream(file);
        ps = new PrintStream(fos);
        System.setOut(ps);
        for (int i = 0; i < discount.length; i++){
            System.out.println("----------------------"+head+"_"+discount[i]+"-----------------------");
            GridWorldGame gw = new GridWorldGame(11, 5, -0.1, discount[i], 0.99);
            System.out.println("-------------Q Learning-------------");
            gw.experimentAndPlotter("QLearning_discount_" + i, 0.3, 0.1);
        }

        System.setOut(dos);
        System.out.println("ALL DONE");
    }

}
