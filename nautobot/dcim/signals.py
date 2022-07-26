import logging

from cacheops import invalidate_obj
from django.contrib.contenttypes.models import ContentType
from django.db.models.signals import post_save, post_delete, pre_delete
from django.db import transaction
from django.dispatch import receiver

from .models import (
    Cable,
    CableEndpoint,
    CablePath,
    Device,
    PathEndpoint,
    PowerPanel,
    Rack,
    RackGroup,
    VirtualChassis,
)

from nautobot.dcim.models.cables import trace_paths

from nautobot.dcim.choices import CableStatusChoices,CableEndpointSideChoices


def create_cablepath(terminations):
    """
    Create CablePaths for all paths originating from the specified set of nodes.

    :param terminations: Iterable of CableTermination objects
    """
    from nautobot.dcim.models import CablePath

    cp = CablePath.from_origin(terminations)
    if cp:
        cp.save()


def rebuild_paths(terminations):
    """
    Rebuild all CablePaths which traverse the specified nodes.
    """
    from nautobot.dcim.models import CablePath

    for obj in terminations:
        cable_paths = CablePath.objects.filter(_nodes__contains=obj)

        with transaction.atomic():
            for cp in cable_paths:
                cp.delete()
                create_cablepath(cp.origins)


#
# Site/rack/device assignment
#


@receiver(post_save, sender=RackGroup)
def handle_rackgroup_site_change(instance, created, **kwargs):
    """
    Update child RackGroups and Racks if Site assignment has changed. We intentionally recurse through each child
    object instead of calling update() on the QuerySet to ensure the proper change records get created for each.
    """
    if not created:
        for rackgroup in instance.get_children():
            rackgroup.site = instance.site
            rackgroup.save()
        for rack in Rack.objects.filter(group=instance).exclude(site=instance.site):
            rack.site = instance.site
            rack.save()
        for powerpanel in PowerPanel.objects.filter(rack_group=instance).exclude(site=instance.site):
            powerpanel.site = instance.site
            powerpanel.save()


@receiver(post_save, sender=Rack)
def handle_rack_site_change(instance, created, **kwargs):
    """
    Update child Devices if Site assignment has changed.
    """
    if not created:
        for device in Device.objects.filter(rack=instance).exclude(site=instance.site):
            device.site = instance.site
            device.save()


#
# Virtual chassis
#


@receiver(post_save, sender=VirtualChassis)
def assign_virtualchassis_master(instance, created, **kwargs):
    """
    When a VirtualChassis is created, automatically assign its master device (if any) to the VC.
    """
    if created and instance.master:
        master = Device.objects.get(pk=instance.master.pk)
        master.virtual_chassis = instance
        if instance.master.vc_position is None:
            master.vc_position = 1
        master.save()


@receiver(pre_delete, sender=VirtualChassis)
def clear_virtualchassis_members(instance, **kwargs):
    """
    When a VirtualChassis is deleted, nullify the vc_position and vc_priority fields of its prior members.
    """
    devices = Device.objects.filter(virtual_chassis=instance.pk)
    for device in devices:
        device.vc_position = None
        device.vc_priority = None
        device.save()


#
# Cables
#


@receiver(trace_paths, sender=CableEndpoint)
def update_connected_endpoints(instance, created, raw=False, **kwargs):
    """
    When a Cable is saved, check for and update its two connected endpoints
    """
    logger = logging.getLogger("nautobot.dcim.cable")
    if raw:
        logger.debug(f"Skipping endpoint updates for imported cable {instance}")
        return

    # Update cable paths if new terminations have been set
    if instance._terminations_modified:
        a_terminations = []
        b_terminations = []
        for cable_endpoint in instance.endpoints.all():
            if cable_endpoint.cable_side == CableEndpointSideChoices.SIDE_A:
                a_terminations.append(cable_endpoint.termination)
            else:
                b_terminations.append(cable_endpoint.termination)
        for nodes in [a_terminations, b_terminations]:
            # Examine type of first termination to determine object type (all must be the same)
            if not nodes:
                continue
            if isinstance(nodes[0], PathEndpoint):
                create_cablepath(nodes)
            else:
                rebuild_paths(nodes)

    # Update status of CablePaths if Cable status has been changed
    elif instance.status != instance._orig_status:
        if instance.status != CableStatusChoices.STATUS_CONNECTED:
            CablePath.objects.filter(_nodes__contains=instance).update(is_active=False)
        else:
            rebuild_paths([instance])


@receiver(post_delete, sender=Cable)
def retrace_cable_paths(instance, **kwargs):
    """
    When a Cable is deleted, check for and update its connected endpoints
    """
    for cablepath in CablePath.objects.filter(_nodes__contains=instance):
        cablepath.retrace()


@receiver(post_delete, sender=CableEndpoint)
def nullify_connected_endpoints(instance, **kwargs):
    """
    When a Cable is deleted, check for and update its two connected endpoints
    """
    model = instance.termination_type.model_class()
    model.objects.filter(pk=instance.termination_id).update(cable=None, cable_side="")
