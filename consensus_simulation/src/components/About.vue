<template>
  <div class="about">
    <h1>{{ msg }}</h1>

    <br>

    <h2>Consensus Sim</h2>
    <p>by Kim, Sungbum</p>

    <br>

    <h2>Final Year Project</h2>
    <p>City University of Hong Kong</p>
    
  </div>
</template>

<script>
export default {
  name: 'About',
  components: {
  },
  data () {
    return {
      msg: 'About'
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h1, h2 {
  font-weight: normal;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
