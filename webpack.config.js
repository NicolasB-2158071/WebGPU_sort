module.exports = {
    mode: 'development',
    entry: './src/index.ts',
    module: {
      rules: [
        {
          test: /\.tsx?$/,
          use: 'ts-loader',
          exclude: /node_modules/
        },
        {
          resourceQuery: /raw/,
          type: 'asset/source'
        }
      ],
    },
    resolve: {
      extensions: ['.tsx', '.ts', '.js'],
    }
  };