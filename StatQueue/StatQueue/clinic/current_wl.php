<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once ("$DOCUMENT_ROOT/functions/statq_fns.php");
$db = db_connect();
do_header();

$clinic = $_SESSION['saved_clinic'];
$clinic->db = db_connect();
$clinic->query_wl();
$clinic->wl_length = $clinic->get_wl_length($clinic->clinicid);
$clinic->display_current_wl();

#line below refreshes the page itself every 30sec. itself: $_SERVER['PHP_SELF']
echo "<meta http-equiv='refresh' content= \"60;URL='/clinic/current_wl.php'\">";
do_footer();
?>