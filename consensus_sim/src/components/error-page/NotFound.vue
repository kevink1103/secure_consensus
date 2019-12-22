<template>
  <b-container>
    <h1>Oops!</h1>
    <p class="msg">
      We can't seem to find the page you're looking for.
    </p>
    <p class="error">
      Error code: 404
    </p>
    <a href="/">
      GO HOME
    </a>
  </b-container>
</template>

<script>
export default {
  name: 'NotFound'
}
</script>

<style scoped>
h1 {
  font-weight: bold;
  font-size: 100px;
  padding-top: 50px;
}
p.msg {
  font-weight: normal;
  font-size: 30px;
  text-align: center;
}
p.error {
  font-weight: normal;
  font-size: 20px;
  text-align: center;
}
a {
  font-weight: normal;
  font-size: 20px;
  text-align: center;
}
</style>
