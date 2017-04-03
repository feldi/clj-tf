(defproject clj-tf "0.1.0"
  :description "Using Tensorflow with Clojure"
  :url "https://github.com/feldi/clj-tf"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"] 
                 [org.tensorflow/libtensorflow "1.1.0-rc1"]
                 [org.clojars.ghaskins/protobuf "3.0.2-2"]
                 [com.google.protobuf/protobuf-java "3.2.0"]] 
  :plugins [[lein-protobuf "0.5.0"]]
  ;; :native-path "resources/jni"
  :jvm-opts ["-Djava.library.path=resources/jni"]
  ;;:java-source-paths ["target/protosrc/org/tensorflow/framework"]
  
  :protobuf-version "3.2.0"
  :protoc "D:/dev/protoc-3.2.0-win32/bin/protoc"
  :proto-path "resources/proto"
  
  :profiles {:dev {:source-paths ["dev-resources"]}}
)