package miningPNMLS;

import java.io.File;
import java.io.IOException;

public class Mine {
	// input a root dir
	// root can be a directory or a file
	public static void traverseFiles(File file, String[] args) {
		if (file.isDirectory()) {
			
			
			// plans to XES 
			if (file.getName().contentEquals("train")) {
				String command = "java -cp xes.jar generate_XES " + file.getPath() + "/";
				
				
				try {
					Runtime.getRuntime().exec(command).waitFor();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
			}
			
			
			
			for (File an_item : file.listFiles()) {
				traverseFiles(an_item, args);
			}
		} else {
			if (file.getName().endsWith(".xes")) {
				System.out.println(file.getPath());
				
				String input_file = file.getPath();
				String output_file = input_file + ".pnml";
				
				String miner = args[0];
				String command = null;
				switch (miner) {
				case "-IM":
					// noiseThreshold: args[2] (float)
					command = String.format("java -cp miner.jar autoMiner -IM %s %s %s", 
							input_file, output_file, args[2]);
					break;
				case "-DFM":
					// noiseThreshold: args[2] (double)
					command = String.format("java -cp miner.jar autoMiner -DFM %s %s %s", 
							input_file, output_file, args[2]);
					break;
				case "-TSM":
					// number of states: args[2] (int)
					// filter the top percentage of event name: args[3] (int)
					// filter the top percentage of label: args[4] (int)
					command = String.format("java -cp miner.jar autoMiner -TSM %s %s %s %s %s", 
							input_file, output_file, args[2], args[3], args[4]);
					break;
				default:
					System.out.println("unknown miner");
				}
				
				try {
					Runtime.getRuntime().exec(command).waitFor();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}
	
	
	public static void main(String[] args) {
		String root_dir = args[1];
		System.out.println("miner starts");
		traverseFiles(new File(root_dir), args);
		//traverseFiles(new File("./gene_data/"));
		System.out.println("mining complete");
	}

}
