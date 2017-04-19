(defproject clj-tf "0.1.5"
  :description "Using Tensorflow with Clojure"
  :url "https://github.com/feldi/clj-tf"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"] 
                 [org.tensorflow/libtensorflow "1.1.0-rc1"]
                 [org.tensorflow/libtensorflow_jni "1.1.0-rc1"]
                 [org.clojars.ghaskins/protobuf "3.0.2-2"]
                 [com.google.protobuf/protobuf-java "3.2.0"]
                 [camel-snake-kebab "0.4.0"]] 
  :plugins [[lein-protobuf "0.5.0"]]
  
  :source-paths ["src/main/clojure"]
  :java-source-paths ["src/main/java"]
  :test-paths ["src/test/clojure"]
  :resource-paths ["src/main/resources"]
  
  ;;:jvm-opts ["-Djava.library.path=resources/jni"]
    
  :protobuf-version "3.2.0"
  :protoc "D:/dev/protoc-3.2.0-win32/bin/protoc"
  :proto-path "src/main/resources/proto"
  
  :profiles {:dev {:source-paths ["dev-resources"]}}
  
  ;; these ones are too big for clojars.org:
  :jar-exclusions  [#"\.dll$" #"\.pb$"]
)