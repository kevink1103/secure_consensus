<template>
  <div class="noise">
    

    <label for="time"><h5>Set time iterations.</h5></label>
    <b-form-input id="time" v-model="time" type="range" min="0" max="100"></b-form-input>
    <div class="mt-1">n: {{ time }}</div>
    

    <br>

    <label for="phi"><h5>Set phi (φ).</h5></label>
    <b-form-input id="phi" v-model="phi" size="mg" placeholder="Enter your name"></b-form-input>
    <div class="mt-1">Range: 0 &lt; φ &lt; 1</div>
  </div>
</template>

<script>
export default {
  name: 'Noise',
  data () {
    return {
      time: '50',
      phi: '0.9'
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>

</style>
