fn main() {
    tonic_build::configure()
        .compile_protos(
            &["proto/mural.proto"], &["proto"])
        .unwrap();
}
