package Project3;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

public class BDRun {

    public static void main(String[] args) throws IOException {
        int[] level = {3};
        double[] reward = {-0.1,1,3,5,10,100};
        double[] discount = {0.99,0.95,0.9,0.8,0.6};
        double[] qInit = {0.3,0.5,1,5,30};
        double[] epsilon = {0.05,0.1,0.3,0.5,0.8};

        OutputStream outs = System.out;
        PrintStream dos = new PrintStream(outs);

        String head = "level";
        System.setOut(dos);
        System.out.println(head);
        new File("./output/"+head).mkdirs();
        File file = new File("./output/"+head+"/out_BD2"); //Your file
        FileOutputStream fos = new FileOutputStream(file);
        PrintStream ps = new PrintStream(fos);
        System.setOut(ps);
        for(int i=0;i<level.length;i++){
            System.out.println("----------------------"+head+"_"+level[i]+"-----------------------");
            BD bd = new BD(5,-0.1,0.99,level[i]);
            System.out.println("-------------Value Iteration-------------");
            bd.valueIteration("./output/"+head+"/BlockDude_"+level[i]);
            System.out.println("-------------Policy Iteration-------------");
            bd.policyIteration("./output/"+head+"/BlockDude_"+level[i]);
        }

        head = "reward";
        System.setOut(dos);
        System.out.println(head);
        new File("./output/"+head).mkdirs();
        file = new File("./output/"+head+"/out_BD"); //Your file
        fos = new FileOutputStream(file);
        ps = new PrintStream(fos);
        System.setOut(ps);
        for(int i=0;i<reward.length;i++){
            System.out.println("----------------------"+head+"_"+reward[i]+"-----------------------");;
            BD bd = new BD(reward[i],-0.1, 0.99,3);
            System.out.println("-------------Value Iteration-------------");
            bd.valueIteration("./output/"+head+"/BlockDude_"+reward[i]);
            System.out.println("-------------Policy Iteration-------------");
            bd.policyIteration("./output/"+head+"/BlockDude_"+reward[i]);
        }

        head = "discount";
        System.setOut(dos);
        System.out.println(head);
        new File("./output/"+head).mkdirs();
        file = new File("./output/"+head+"/out_BD"); //Your file
        fos = new FileOutputStream(file);
        ps = new PrintStream(fos);
        System.setOut(ps);
        for (int i=0; i < discount.length; i++){
            System.out.println("----------------------"+head+"_"+discount[i]+"-----------------------");
            BD bd = new BD(5, -0.1, discount[i],3);
            System.out.println("-------------Value Iteration-------------");
            bd.valueIteration("./output/"+head+"/BlockDude_"+discount[i]);
            System.out.println("-------------Policy Iteration-------------");
            bd.policyIteration("./output/"+head+"/BlockDude_"+discount[i]);
        }

        head = "qInit";
        System.setOut(dos);
        System.out.println(head);
        new File("./output/"+head).mkdirs();
        file = new File("./output/"+head+"/out_BD"); //Your file
        fos = new FileOutputStream(file);
        ps = new PrintStream(fos);
        System.setOut(ps);
        for(int i=0;i<qInit.length;i++){
            System.out.println("----------------------"+head+"_"+qInit[i]+"-----------------------");
            BD bd = new BD(5, -0.1, 0.99,3);
            System.out.println("-------------Q Learning-------------");
            bd.experimentAndPlotter(qInit[i]);
        }


        head = "epsilon";
        System.setOut(dos);
        System.out.println(head);
        new File("./output/"+head).mkdirs();
        file = new File("./output/"+head+"/out_BD"); //Your file
        fos = new FileOutputStream(file);
        ps = new PrintStream(fos);
        System.setOut(ps);
        for(int i=0;i<epsilon.length;i++){
            System.out.println("----------------------"+head+"_"+epsilon[i]+"-----------------------");
            BD bd = new BD(5, -0.1, 0.99,1);
            System.out.println("-------------Q Learning-------------");
            bd.epsilonComparison("./output/"+head+"/BlockDude_"+epsilon[i], 0.3, 1, epsilon[i]);
        }
        System.setOut(dos);
        System.out.println("ALL DONE");

    }

}
