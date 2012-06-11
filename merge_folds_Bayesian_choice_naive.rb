require "rubygems"
require "faster_csv"
require "find"

Find.find('./results') do |path|
	if FileTest.directory?(path)
		filesInDirectory = Dir.glob("#{path}/*") # Get list of files in directory

		# Add files with filename matching /graph.csv.[0-9]+/ regex to list
		filesToProcess = Array.new
		filesInDirectory.each do |file|
			if file =~ /Bayesian.choice.naive.graph.csv.[0-9]+/ 
				filesToProcess << file
			end
		end

		# Process list of files that matched regex in current directory
		if filesToProcess.count > 0
			puts "MERGING FILES:"
			puts filesToProcess
			puts ""

			# Separates files for folds into buckets by test
			dataFiles = Array.new
			filesToProcess.each { |f| dataFiles << f.clone }
			dataFiles.each { |f| f.slice!(/[.]*(.)[0-9]+$/) } # removes fold extension
			dataFiles.uniq! # removes duplicates

			dataFiles.each { |f| f.slice!(/\/[0-9-]+\/[A-Za-z.]+$/) } # removes end (to base dir)
			
			# Infer output file name
			dataFileName = dataFiles[0]
			i = dataFileName.rindex("/") + 1
			dataFileName = dataFileName[i, dataFileName.length - i]
			#puts dataFileName

			# Infer output root folder
			dataRootPath = dataFiles[0]
			dataRootPath.slice!(/[A-Za-z0-9=;,]+$/) # removes end (to base dir)
			#puts dataRootPath

			# Construct output file path
			dataFilePath = dataRootPath + dataFileName + "Bayesian.choice.naive.csv"
			puts dataFilePath

			# Set output file
			outFile = dataFilePath

			# Merges the files in each bucket
			files = Array.new
			filesToProcess.each { |f| if not f.index(dataFiles[0]).nil? then files << f end }
			files.sort!

			k = files.size.to_f # The number of folds this test has

			mdata = Array.new
			filesToProcess.each do |f|
				data = FasterCSV.read(f, {:col_sep => "\t"})

				# Adds data to summation matrix
				data.each_index do |i|
					row = data[i] # Stores row array

					# check if this is the first time we're encountering this row
					if mdata.size == i then
						# Creates a new row array to store results
						mrow = Array.new
						mrow << row[0].to_i # add row number

						mrow << row[3].to_f # add minimum classification error
						mrow << row[3].to_f # add average classification error
						mrow << row[3].to_f # add maximum classification error

						mrow << row[4].to_f # add minimum log loss
						mrow << row[4].to_f # add average log loss
						mrow << row[4].to_f # add maximum log loss
						mdata << mrow
					elsif
						# Adds results to existing row array
						mrow = mdata[i]

						# Update minimum classification error if necessary
						if row[3].to_f < mrow[1] then
							mrow[1] = row[3].to_f # min error
						end

						# Add to classification error average
						mrow[2] = mrow[2].to_f + row[3].to_f # average error

						# Update maximum classification error if necessary
						if row[3].to_f > mrow[3] then
							mrow[3] = row[3].to_f # min error
						end

						# Update minimum loss if necessary
						if row[4].to_f < mrow[4] then
							mrow[4] = row[4].to_f # min error
						end

						# Add to loss average
						mrow[5] = mrow[5].to_f + row[4].to_f # average error

						# Update maximum loss if necessary
						if row[4].to_f > mrow[6] then
							mrow[6] = row[4].to_f # min error
						end
					end
				end
			end

			# Calculates average
			mdata.each_index do |i|
				mrow = mdata[i]
				mrow[2] = mrow[2].to_f / k.to_f
				mrow[5] = mrow[5].to_f / k.to_f
			end

			# Writes merged data to new file (overwrites if exists)
			File.delete(outFile) if File::exists?(outFile)
			FasterCSV.open(outFile, "w", {:col_sep => "\t"}) do |csv|
				mdata.each { |r| csv << r }
			end

			# Adds fold data to *.tar.gz archive
			#archiveFile = outFile.clone
			#archiveFile.slice!(/[.]*dat$/)
			#contents = ""
			#files.each { |f| contents.concat(inPath + f + ' ') }
			#str = "tar -pczf #{archiveFile}.`date +\"%Y-%m-%d\"`.tar.gz #{contents}"
			#system(str)
		end
	end
end
