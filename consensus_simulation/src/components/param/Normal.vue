<template>
  <div class="normal">
    <label for="time"><h5>Set time iterations.</h5></label>
    <b-form-input id="time" v-model="time" type="range" min="0" max="100"></b-form-input>
    <div class="mt-1">n: {{ time }}</div>
  </div>
</template>

<script>
export default {
  name: 'Normal',
  data () {
    return {
      time: '50'
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>

</style>
